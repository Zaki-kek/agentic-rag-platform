# agentic-rag-platform

A production-style **RAG assistant**: upload documents, ask questions, get answers
**grounded with citations**. Built with FastAPI, pgvector and pluggable LLM
providers, with a clean factory-based architecture and an offline mode so the whole
stack runs and is testable with **no API keys and no database**.

> Portfolio / reference project. It distills patterns I use in real shipped products
> (multi-provider LLM routing, document ingestion, a Telegram delivery layer) into a
> neutral, self-contained service.

---

## Highlights

- **RAG over your documents** - PDF / DOCX / TXT / MD ingestion (PyMuPDF, python-docx) -> chunking -> embeddings -> vector search -> grounded answer with citations.
- **Pluggable everything (Factory + Registry)** - LLM provider (`echo`, `openai`, `anthropic`, `gigachat`), embedder (`hash`, `openai`), vector store (`memory`, `pgvector`). Swap via env, no code change.
- **Offline-first** - deterministic `echo` LLM + `hash` embedder + in-memory store mean `make test` and `make run` work with zero secrets.
- **Production shape** - async FastAPI, typed pydantic contracts, layered design, Docker Compose with pgvector, tests, ruff + mypy, CI-ready.
- **Grounded, not hallucinated** - answers cite the exact chunks they used.
- **Resumable generation pipeline** - a checkpointed multi-stage orchestrator with quality gates and **deterministic computation** (numbers are computed in Python; the LLM only narrates them, and a gate proves it did not alter them), plus a job-status API and a payment stub.
- **Tool-using agent** - a dependency-free ReAct loop with a tool registry (safe calculator, RAG retrieval), LangGraph-ready, exposed at `/agent`.
- **Evals, tracing, guardrails** - a lightweight RAG eval harness (keyword-recall / grounding / numbers-preserved over a golden set), vendor-neutral tracing (no-op by default, optional Langfuse), and PII redaction + citation validation.
- **Streaming, Telegram, graph orchestration** - SSE token streaming (`/chat/stream`), a transport-agnostic Telegram delivery adapter (`/telegram/message`, lazy aiogram), a dependency-free LangGraph-style state-graph runner, a rate-limit-aware retrying provider wrapper, and weighted multi-stage job progress.

## Architecture

See [docs/architecture.md](docs/architecture.md). In short:

```
api (HTTP) -> services (orchestration) -> rag (extract/chunk/embed/store) + llm (providers)
```

## Quickstart

### Local (offline, no keys)

```bash
pip install -e ".[llm,dev]"
make run            # http://localhost:8000/docs
```

```bash
# ingest a document, then ask about it
curl -F "file=@docs/architecture.md" http://localhost:8000/documents
curl -s http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d '{"message":"What design patterns does this project use?"}' | jq
```

### Full stack (Docker + pgvector)

```bash
docker compose up --build      # api on :8000, pgvector on :5432
```

### Real models

```bash
cp .env.example .env
# set LLM_PROVIDER=openai, EMBEDDER=openai, OPENAI_API_KEY=sk-...
make run
```

## Configuration

| Env | Default | Options |
|-----|---------|---------|
| `LLM_PROVIDER` | `echo` | `echo`, `openai`, `anthropic`, `gigachat` |
| `EMBEDDER` | `hash` | `hash`, `openai` |
| `VECTOR_STORE` | `memory` | `memory`, `pgvector` |
| `TRACER` | `none` | `none`, `memory`, `langfuse` |
| `LLM_MAX_CONCURRENCY` | `8` | provider concurrency cap |
| `TOP_K` | `4` | retrieval depth |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | chunking |

Full list in [.env.example](.env.example).

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/documents` | upload + ingest a document |
| `POST` | `/chat` | ask a question, get a grounded answer + citations |
| `POST` | `/chat/stream` | stream the answer as Server-Sent Events |
| `POST` | `/jobs` | run a multi-stage generation job (compute -> draft -> gate) |
| `GET`  | `/jobs/{id}` | poll job status / result |
| `POST` | `/payments` | create a payment (stub) |
| `POST` | `/payments/{id}/confirm` | confirm a payment (webhook-style) |
| `POST` | `/agent` | answer via the tool-using agent (calculator, retrieval) |
| `POST` | `/guardrails/redact` | redact PII (emails, phones, cards, IPs) |
| `POST` | `/guardrails/citations` | validate `[n]` citation markers |
| `POST` | `/telegram/message` | simulate a Telegram message through the assistant |
| `GET`  | `/health` | liveness + active config |

Interactive docs at `/docs`.

## Development

```bash
make test       # pytest (offline, no secrets)
make lint       # ruff
make typecheck  # mypy
```

## Roadmap

Shipped:
- [x] Checkpointed multi-stage generation orchestrator with recovery
- [x] Quality gates + deterministic computation (verified-numbers pattern)
- [x] Job-status API and payment stub
- [x] Tool-using agent (ReAct loop + tool registry; LangGraph-ready)
- [x] Vendor-neutral tracing (no-op default, optional Langfuse)
- [x] RAG eval harness over a golden set
- [x] Guardrails: PII redaction + citation validation
- [x] SSE streaming, Telegram delivery adapter, weighted job progress
- [x] Rate-limit-aware retrying provider wrapper
- [x] Dependency-free state-graph runner (LangGraph-style)

Next:
- [ ] Swap the in-house graph/agent runner for LangGraph in production
- [ ] Native provider token streaming (vs chunked SSE)

## Tech

Python 3.11+, FastAPI, pydantic v2, numpy, PyMuPDF, python-docx, asyncpg + pgvector,
OpenAI / Anthropic SDKs, a ReAct agent loop, a RAG eval harness, vendor-neutral tracing
(optional Langfuse), pytest, ruff, mypy, Docker.
