# Architecture

## Request flow

```mermaid
flowchart LR
    U[Client] -->|POST /documents| API[FastAPI]
    U -->|POST /chat| API
    API --> PIPE[RAG pipeline]
    PIPE --> EX[Extract: PyMuPDF / python-docx]
    PIPE --> CH[Chunk + overlap]
    PIPE --> EMB[Embedder factory: hash | openai]
    EMB --> VS[(Vector store: memory | pgvector)]
    API --> SVC[Chat service]
    SVC --> VS
    SVC --> LLM[LLM factory: echo | openai | anthropic | gigachat]
    SVC -->|answer + citations| U
```

## Design notes

- **Factory + Registry** for LLM providers, embedders and vector stores: swapping
  any backend is a config change, not a code change.
- **Offline-first defaults** (`echo` / `hash` / `memory`): the whole stack runs and
  is fully testable with no API keys and no database. Production swaps in real
  providers and pgvector via environment variables.
- **Clean layering**: `api` (HTTP) -> `services` (orchestration) -> `rag` + `llm`
  (capabilities). Schemas in `schemas.py` are the public contract.
- **Grounded answers**: every reply carries citations to the retrieved chunks.

## Extension points (roadmap)

- Agentic layer (LangGraph) with tool-calling on top of `ChatService`.
- Tracing (Langfuse) around provider + retrieval calls.
- Eval harness (ragas) over a golden question set.
- Streaming responses (SSE) and a Telegram adapter reusing the same service.
