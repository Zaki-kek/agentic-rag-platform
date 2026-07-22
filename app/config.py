"""Application configuration (immutable, environment-driven)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, loaded from environment / .env.

    Defaults are chosen so the service runs fully offline (no API keys,
    no database) out of the box: provider=echo, embedder=hash, store=memory.
    Switch to real providers via environment variables.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_name: str = "agentic-rag-platform"
    log_level: str = "INFO"

    # LLM provider: echo | openai | anthropic | gigachat
    llm_provider: str = "echo"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gigachat_credentials: str | None = None
    llm_max_concurrency: int = 8
    # Per-call wall-clock budget (seconds) for a single LLM generate attempt.
    llm_timeout_seconds: float = 60.0

    # Embeddings: hash | openai
    embedder: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    hash_embedding_dim: int = 256

    # Vector store: memory | pgvector
    vector_store: str = "memory"
    db_dsn: str = "postgresql://postgres:postgres@db:5432/assistant"

    # Observability: none | memory | langfuse
    tracer: str = "none"

    # Agent token budget across a single run (rough estimate: len(text)/4 per
    # LLM call). 0 disables the budget (steps are still capped by max_steps).
    agent_max_tokens: int = 0

    # Generation checkpoint store: memory | file
    checkpoint_store: str = "memory"
    checkpoint_dir: str = ".checkpoints"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    @property
    def is_offline(self) -> bool:
        """True when no external network dependency is configured."""
        return self.llm_provider == "echo" and self.embedder == "hash"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
