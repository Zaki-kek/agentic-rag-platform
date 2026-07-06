"""Pydantic request/response schemas (the public API contract)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A single user question to answer over the indexed documents."""

    message: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override retrieval depth")


class Citation(BaseModel):
    """A retrieved chunk that supports the answer."""

    document: str
    chunk_id: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    """Model answer plus the sources it was grounded on."""

    answer: str
    citations: list[Citation]
    provider: str


class IngestResponse(BaseModel):
    """Result of ingesting a document into the vector store."""

    document: str
    chunks: int


class HealthResponse(BaseModel):
    """Liveness / configuration snapshot."""

    status: str = "ok"
    llm_provider: str
    embedder: str
    vector_store: str
