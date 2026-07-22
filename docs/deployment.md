# Deployment

How to run agentic-rag-platform in production. The service is a stateless async
FastAPI app; the only stateful dependency is Postgres (pgvector) when you turn on
persistent vector search. Everything else is process-local.

## Deployment shapes

| Shape | LLM / embeddings | Vector store | External deps | Cost |
|-------|------------------|--------------|---------------|------|
| Offline | `echo` / `hash` | `memory` | none | $0 |
| Real models, no DB | `openai` / `anthropic` / `gigachat` | `memory` | provider API | per-token |
| Full production | real provider | `pgvector` | provider API + Postgres | per-token + DB host |

The offline shape is the default and needs no secrets and no database - see the
[Cost controls](../README.md#cost-controls) section. Use it for CI, smoke tests
and demos.

## Environment variables

Copy `.env.example` to `.env` and override what you need. The defaults run fully
offline. For a real deployment the variables that actually matter are:

### Required for real models

| Env | When required |
|-----|---------------|
| `LLM_PROVIDER` | set to `openai`, `anthropic` or `gigachat` (default `echo` needs nothing) |
| `OPENAI_API_KEY` | when `LLM_PROVIDER=openai` **or** `EMBEDDER=openai` |
| `ANTHROPIC_API_KEY` | when `LLM_PROVIDER=anthropic` |
| `GIGACHAT_CREDENTIALS` | when `LLM_PROVIDER=gigachat` |
| `EMBEDDER` | set to `openai` for real embeddings (default `hash` needs nothing) |

Missing or malformed provider config surfaces as a `provider_config_error` (HTTP
500) with the uniform error body, not a silent fallback.

### Required for persistent vector search

| Env | When required |
|-----|---------------|
| `VECTOR_STORE` | set to `pgvector` (default `memory` is process-local, lost on restart) |
| `DB_DSN` | Postgres DSN, e.g. `postgresql://user:pass@host:5432/assistant` |

The pgvector store creates its table and an HNSW cosine index (`vector_cosine_ops`)
on startup, so a fresh database is bootstrapped automatically. The Postgres image
must ship the `vector` extension (`pgvector/pgvector:pg16`).

### Common tuning knobs

`LLM_MAX_CONCURRENCY`, `LLM_TIMEOUT_SECONDS`, `TOP_K`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `AGENT_MAX_TOKENS`, `TRACER`, `CHECKPOINT_STORE`. Full list with
comments in [.env.example](../.env.example); cost implications in
[Cost controls](../README.md#cost-controls).

Never commit real secrets. Keep `.env` out of the image (it is gitignored and
`.dockerignore`d); inject secrets at runtime via your orchestrator.

## Rolling it out

### Docker Compose (single host)

The base `docker-compose.yml` builds the API and starts pgvector for local full-stack
runs. For production, layer the override on top:

```bash
cp .env.example .env
# edit .env: LLM_PROVIDER=openai, EMBEDDER=openai, OPENAI_API_KEY=sk-..., etc.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

The prod override (`docker-compose.prod.yml`):

- drops the host port publish on Postgres (DB reachable only on the internal
  Compose network, not the public interface);
- makes `DB_DSN` and every provider setting come from `.env` / the environment
  rather than hardcoded dev values;
- sets `restart: unless-stopped` and pins container log rotation so a long-running
  host does not fill its disk with logs.

Put a TLS-terminating reverse proxy (nginx, Caddy, a cloud LB) in front of the API;
the container itself speaks plain HTTP on `:8000`.

### Single container (managed platform)

The image is self-contained (multi-stage build, non-root `app` user, no build
toolchain in the runtime layer). On a PaaS you only need the API container plus a
managed Postgres:

```bash
docker build -t agentic-rag-platform .
docker run -p 8000:8000 --env-file .env agentic-rag-platform
```

## Health checks

The app exposes `GET /health` (also `GET /v1/health`), which returns `200` with the
active provider / embedder / vector-store config. Use it as both the liveness and
readiness probe.

- The `Dockerfile` already declares a `HEALTHCHECK` that hits `/health` using the
  stdlib (no `curl` needed in the image).
- Kubernetes: point `livenessProbe` and `readinessProbe` at `/health`.
- Prometheus-style metrics are at `GET /metrics` (request counter, text exposition
  format).

`/health` does not open a DB connection - it reflects configuration, not
dependency reachability. If you need a deep DB check, add one behind a separate
path; a failing pgvector connection currently surfaces on the first `/documents`
or `/chat` call instead.

## What is free ($0)

In the default offline shape - `LLM_PROVIDER=echo`, `EMBEDDER=hash`,
`VECTOR_STORE=memory`, `TRACER=none` - the service makes **no** outbound network
calls and needs **no** database. The full test suite and `make run` work with zero
secrets and zero cost. Everything that costs money (LLM tokens, embedding tokens,
a hosted Postgres) is opt-in via the environment variables above.
