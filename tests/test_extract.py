"""Unit tests for document text extraction."""

from __future__ import annotations

import pytest

from app.core import UnsupportedFileTypeError
from app.rag.extract import extract_text


def test_extract_txt() -> None:
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_extract_markdown() -> None:
    assert extract_text("readme.md", b"# Title\ntext") == "# Title\ntext"


def test_unsupported_type_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("image.png", b"\x89PNG")


def test_extract_docx_roundtrip() -> None:
    docx = pytest.importorskip("docx")  # python-docx
    import io

    document = docx.Document()
    document.add_paragraph("first paragraph")
    document.add_paragraph("second paragraph")
    buffer = io.BytesIO()
    document.save(buffer)

    text = extract_text("doc.docx", buffer.getvalue())
    assert "first paragraph" in text
    assert "second paragraph" in text
