"""/documents route: ingest an uploaded document into the vector store."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile

from app.schemas import IngestResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile) -> IngestResponse:
    pipeline = request.app.state.pipeline
    data = await file.read()
    count = await pipeline.ingest(file.filename or "upload", data)
    return IngestResponse(document=file.filename or "upload", chunks=count)
