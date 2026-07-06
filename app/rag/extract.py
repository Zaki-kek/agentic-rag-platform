"""Text extraction from uploaded documents.

Mirrors the document pipeline pattern (PyMuPDF for PDF, python-docx for DOCX),
generalised to a neutral ingestion step. Heavy parsers are lazy-imported.
"""

from __future__ import annotations

from pathlib import Path

from app.core import UnsupportedFileTypeError

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def extract_text(filename: str, data: bytes) -> str:
    """Return plain text for a supported document, dispatching by extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED)}")


def _from_pdf(data: bytes) -> str:
    import fitz  # PyMuPDF, lazy-imported

    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _from_docx(data: bytes) -> str:
    import io

    from docx import Document  # python-docx, lazy-imported

    document = Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)
