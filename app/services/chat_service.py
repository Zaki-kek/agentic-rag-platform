"""Chat service: grounds an LLM answer in retrieved document chunks."""

from __future__ import annotations

from app.llm.base import LLMProvider, Message
from app.observability import NoOpTracer, Tracer
from app.rag.pipeline import RagPipeline
from app.rag.store import Hit
from app.schemas import ChatResponse, Citation

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the context is insufficient, say so plainly. Be concise and cite "
    "sources by their [n] markers."
)

_PREVIEW_CHARS = 240


class ChatService:
    """Compose a grounded answer from retrieval + an LLM provider."""

    def __init__(
        self,
        pipeline: RagPipeline,
        provider: LLMProvider,
        provider_name: str,
        default_k: int = 4,
        tracer: Tracer | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._provider = provider
        self._provider_name = provider_name
        self._default_k = default_k
        self._tracer = tracer or NoOpTracer()

    async def answer(self, message: str, top_k: int | None = None) -> ChatResponse:
        k = top_k or self._default_k
        with self._tracer.span("chat.answer", provider=self._provider_name, top_k=k) as span:
            hits = await self._pipeline.retrieve(message, k)
            span.set(retrieved=len(hits))
            messages = self._build_messages(message, hits)
            reply = await self._provider.generate(messages)
            return ChatResponse(
                answer=reply,
                citations=[self._to_citation(h) for h in hits],
                provider=self._provider_name,
            )

    def _build_messages(self, question: str, hits: list[Hit]) -> list[Message]:
        if hits:
            context = "\n\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
        else:
            context = "(no relevant context found)"
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _to_citation(hit: Hit) -> Citation:
        preview = hit.text[:_PREVIEW_CHARS] + ("..." if len(hit.text) > _PREVIEW_CHARS else "")
        return Citation(document=hit.document, chunk_id=hit.chunk_id, score=round(hit.score, 4), preview=preview)
