"""Application factory and ASGI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import Agent, CalculatorTool, RetrievalTool, ToolRegistry
from app.api import (
    agent_router,
    chat_router,
    documents_router,
    guardrails_router,
    jobs_router,
    payments_router,
    stream_router,
    telegram_router,
)
from app.config import Settings, get_settings
from app.core import configure_logging, get_logger, register_exception_handlers
from app.generation import GenerationOrchestrator, InMemoryCheckpointStore, build_report_pipeline
from app.llm import build_provider
from app.llm.ratelimit import RateLimitedProvider
from app.observability import build_tracer
from app.payments import StubPaymentProvider
from app.rag import RagPipeline, build_embedder, build_store
from app.schemas import HealthResponse
from app.services import ChatService
from app.telegram import AssistantFacade

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app with all dependencies wired up."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    provider = RateLimitedProvider(build_provider(settings), max_concurrency=settings.llm_max_concurrency)
    tracer = build_tracer(settings.tracer)
    pipeline = RagPipeline(embedder, store, settings.chunk_size, settings.chunk_overlap)
    chat_service = ChatService(pipeline, provider, settings.llm_provider, settings.top_k, tracer)

    checkpoint_store = InMemoryCheckpointStore()
    orchestrator = GenerationOrchestrator(build_report_pipeline(provider), checkpoint_store)
    payment_provider = StubPaymentProvider()

    tools = ToolRegistry()
    tools.add(CalculatorTool()).add(RetrievalTool(pipeline.retrieve))
    agent = Agent(provider, tools)

    async def _answer_fn(text: str) -> tuple[str, list[str]]:
        response = await chat_service.answer(text)
        return response.answer, [c.preview for c in response.citations]

    telegram_facade = AssistantFacade(_answer_fn)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await store.init()
        logger.info("Vector store ready: %s", settings.vector_store)
        yield
        await store.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.chat_service = chat_service
    app.state.generation_orchestrator = orchestrator
    app.state.checkpoint_store = checkpoint_store
    app.state.payment_provider = payment_provider
    app.state.agent = agent
    app.state.tracer = tracer
    app.state.telegram_facade = telegram_facade

    register_exception_handlers(app)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(stream_router)
    app.include_router(jobs_router)
    app.include_router(payments_router)
    app.include_router(agent_router)
    app.include_router(guardrails_router)
    app.include_router(telegram_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return HealthResponse(
            llm_provider=settings.llm_provider,
            embedder=settings.embedder,
            vector_store=settings.vector_store,
        )

    return app


app = create_app()
