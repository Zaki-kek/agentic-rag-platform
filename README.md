# agentic-rag-platform

[![CI](https://github.com/Zaki-kek/agentic-rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Zaki-kek/agentic-rag-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
- **Indexed vector search** - the pgvector store creates an HNSW ANN index (`vector_cosine_ops`) on init, so `<=>` cosine search uses an Index Scan rather than a full Seq Scan. This is verified by a docker-backed integration test (`tests/integration/test_pgvector_index.py`, `EXPLAIN ANALYZE` asserts `Index Scan`); it is skipped unless `DB_DSN` points at a live pgvector instance, so the offline suite stays green.
- **Production shape** - async FastAPI, typed pydantic contracts, layered design, Docker Compose with pgvector, tests, ruff + mypy, CI-ready.
- **Grounded, not hallucinated** - answers cite the exact chunks they used.
- **Resumable generation pipeline** - a checkpointed multi-stage orchestrator with quality gates and **deterministic computation** (numbers are computed in Python; the LLM only narrates them, and a gate proves it did not alter them), plus a job-status API and a payment stub.
- **Tool-using agent** - a dependency-free ReAct loop with a tool registry (safe calculator, RAG retrieval), a step cap plus an optional token budget (rough `chars/4` estimate via `AGENT_MAX_TOKENS`), a tracer span per step, LangGraph-ready, exposed at `/agent`.
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
| `CHECKPOINT_STORE` | `memory` | `memory`, `file` (persist job state to `CHECKPOINT_DIR` as JSON) |
| `AGENT_MAX_TOKENS` | `0` | per-run token budget (rough `chars/4` estimate); `0` = unlimited |

Full list in [.env.example](.env.example).

## Cost controls

Cost is opt-in. The knobs below are the levers for keeping spend and load
predictable; no dollar figures are quoted here because they depend on your
provider, model and traffic - these are the mechanisms, not a price list.

- **Offline mode = $0.** The default `LLM_PROVIDER=echo`, `EMBEDDER=hash`,
  `VECTOR_STORE=memory`, `TRACER=none` makes **no** outbound calls and needs
  **no** database. CI, smoke tests, demos and `make run` cost nothing. Everything
  billable is opt-in via env.
- **`LLM_MAX_CONCURRENCY`** - caps in-flight provider requests. Lower it to bound
  peak spend rate and stay under provider rate limits; raise it for throughput.
- **`LLM_TIMEOUT_SECONDS`** - cancels a slow call instead of letting it hang (and
  keep a connection - and potential retries - alive).
- **`TOP_K`** - fewer retrieved chunks means a shorter prompt, so fewer input
  tokens per `/chat` call. Trade recall for cost.
- **`CHUNK_SIZE` / `CHUNK_OVERLAP`** - larger chunks and less overlap mean fewer
  embeddings per document (cheaper ingest) and fewer, larger retrieved chunks;
  smaller chunks improve precision at higher embed and prompt cost.
- **`AGENT_MAX_TOKENS`** - a per-run token budget for the agent (rough `chars/4`
  estimate). Set it to hard-stop a runaway ReAct loop before it burns tokens;
  `0` = unlimited (still capped by `max_steps`).
- **Embedder choice** - `hash` embeddings are free and offline; switch to
  `openai` only when you need real semantic search.
- **Reuse the vector store as a cache** - documents are embedded once at ingest
  and reused across every query, so retrieval itself costs no embedding tokens
  (only the query is embedded). Persist with `VECTOR_STORE=pgvector` so a restart
  does not force a re-ingest.

See [docs/deployment.md](docs/deployment.md) for what is free vs billable per
deployment shape.

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

### Versioning

Every router is dual-mounted: the unversioned paths above stay stable, and the
same endpoints are also served under a `/v1` prefix (e.g. `/v1/chat`,
`/v1/health`). New breaking changes get a new prefix without touching `/v1`.

### Error shape

Errors return a uniform JSON body:

```json
{ "error": "human-readable message", "code": "machine_identifier" }
```

Codes include `validation_error` (422), `unsupported_media_type` (415),
`provider_config_error` (500) and `internal_error` (500 fallback).

## Development

```bash
make test       # pytest (offline, no secrets)
make lint       # ruff
make typecheck  # mypy
```

## Benchmarks

Offline micro-benchmarks (`time.perf_counter`, no network) over the hash
embedder and in-memory store:

```bash
python -m app.bench --repeat 3
```

Representative numbers from one run (absolute timings vary by machine; the
load-bearing figure is the **call count**, which is deterministic):

| benchmark | metric | value | note |
| --- | --- | --- | --- |
| embed throughput (hash, batched) | texts/sec | ~85,000 | offline hash - batching does not speed this up |
| embed throughput (hash, per-text) | texts/sec | ~105,000 | same Python loop, slightly faster; the win is on a networked provider |
| embed wall-clock (delay-mock, batched) | ms (100 texts) | ~6.5 | **4** simulated calls @ 1.0ms each |
| embed wall-clock (delay-mock, per-text) | ms (100 texts) | ~118 | **100** simulated calls @ 1.0ms each |
| retrieve latency p50 | ms | ~0.3 | 500 vectors, top-5, in-memory cosine |
| retrieve latency p95 | ms | ~0.4 | 500 vectors, top-5, in-memory cosine |
| ingest latency (synthetic doc) | ms | ~0.5 | 2280 bytes -> extract+chunk+embed+store |

The batching win is measured on a **simulated network latency** (`DelayEmbedder`
sleeps once per `embed` call): batching 100 texts at `batch_size=32` collapses
100 round-trips into 4, so wall-clock drops ~18x here. On the offline hash
embedder there is no such win - it is the same Python loop - and the report says
so rather than manufacturing one.

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
- [x] Small demo: a linear probe (pure-numpy logistic regression) over the embeddings on a toy dataset (`python -m app.ml`)

Next:
- [ ] Swap the in-house graph/agent runner for LangGraph in production
- [ ] Native provider token streaming (vs chunked SSE)

## Tech

Python 3.11+, FastAPI, pydantic v2, numpy, PyMuPDF, python-docx, asyncpg + pgvector,
OpenAI / Anthropic SDKs, a ReAct agent loop, a RAG eval harness, vendor-neutral tracing
(optional Langfuse), pytest, ruff, mypy, Docker.
