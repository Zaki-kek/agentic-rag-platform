"""Tests for the offline batch-ingest CLI (``python -m app.rag``).

These pin the batch-ingest contract that makes the CLI safe to point at a real
directory: it counts successful ingests, isolates a single broken file behind a
per-file ``except`` (a garbage ``.pdf`` fails without aborting the run),
distinguishes a validly-empty file from a failure, persists a content-hash
checkpoint, and skips already-ingested documents (without re-embedding) on a
second run. Everything is offline - hash embedder + in-memory store, no
network.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag import ingest_cli
from app.rag.ingest_cli import (
    IngestReport,
    _ingest_files,
    _load_checkpoint,
    _save_checkpoint,
    main,
)
from app.rag.pipeline import RagPipeline
from app.rag.store import InMemoryVectorStore

# A byte string that is a valid PDF *name* but not valid PDF *content*, so the
# lazy PyMuPDF parser raises FileDataError (a RuntimeError, NOT an AppError) -
# exactly the "one bad file in a batch" case the CLI must survive.
_GARBAGE_PDF = b"this is not a real pdf %%%% just some bytes"


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def test_ingests_three_text_files_then_skips_on_rerun(tmp_path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", b"The Eiffel Tower is in Paris and stands 330 metres tall.")
    _write(docs / "b.txt", b"Photosynthesis converts sunlight into chemical energy in plants.")
    _write(docs / "c.md", b"Water boils at 100 degrees Celsius at sea level.")
    ckpt = tmp_path / "ckpt"

    # First run: all three ingest cleanly.
    exit_1 = main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "json"])
    report_1 = json.loads(capsys.readouterr().out)
    assert exit_1 == 0
    assert report_1["ingested"] == 3
    assert report_1["skipped"] == 0
    assert report_1["total_chunks"] >= 3

    # Second run over the same directory: every file's hash is in the
    # checkpoint, so all three are skipped and nothing is re-ingested.
    exit_2 = main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "json"])
    report_2 = json.loads(capsys.readouterr().out)
    assert exit_2 == 0
    assert report_2["skipped"] == 3
    assert report_2["ingested"] == 0
    assert report_2["total_chunks"] == 0


async def test_rerun_does_not_reembed_skipped_files(tmp_path) -> None:
    # White-box check of the "no re-embed on rerun" guarantee using a counting
    # embedder: after the checkpoint is populated, a second pass with the SAME
    # hashes must skip before any embed call, so the counter stays at 0.
    class _CountingEmbedder:
        def __init__(self) -> None:
            self.dim = 256
            self.calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[0.0] * self.dim for _ in texts]

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    _write(a, b"The Eiffel Tower is in Paris and stands 330 metres tall.")
    _write(b, b"Photosynthesis converts sunlight into chemical energy in plants.")
    paths = [a, b]

    seen: set[str] = set()
    first_embedder = _CountingEmbedder()
    first_pipeline = RagPipeline(first_embedder, InMemoryVectorStore())
    first = await _ingest_files(paths, first_pipeline, seen)
    assert len(first.ingested) == 2
    assert first_embedder.calls > 0  # embedded on the first pass
    assert len(seen) == 2  # both hashes now recorded

    # Second pass with the populated hash set: everything is skipped and the
    # fresh embedder is never called.
    second_embedder = _CountingEmbedder()
    second_pipeline = RagPipeline(second_embedder, InMemoryVectorStore())
    second = await _ingest_files(paths, second_pipeline, seen)
    assert len(second.skipped) == 2
    assert len(second.ingested) == 0
    assert second_embedder.calls == 0  # nothing re-embedded


def test_garbage_pdf_isolated_as_failed_run_does_not_crash(tmp_path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", b"The Eiffel Tower is in Paris and stands 330 metres tall.")
    _write(docs / "b.txt", b"Water boils at 100 degrees Celsius at sea level.")
    _write(docs / "broken.pdf", _GARBAGE_PDF)  # real FileDataError on parse
    ckpt = tmp_path / "ckpt"

    exit_code = main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "json"])
    report = json.loads(capsys.readouterr().out)

    # The batch survives the broken file: two ingest, one fails, exit stays 0.
    assert exit_code == 0
    assert report["ingested"] == 2
    assert report["failed"] == 1
    assert report["failed_sources"][0]["source"].endswith("broken.pdf")
    assert "FileDataError" in report["failed_sources"][0]["reason"]


def test_empty_text_file_is_empty_not_failed(tmp_path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", b"The Eiffel Tower is in Paris and stands 330 metres tall.")
    _write(docs / "empty.txt", b"")  # valid, but yields zero chunks
    ckpt = tmp_path / "ckpt"

    exit_code = main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["empty"] == 1
    assert report["failed"] == 0  # an empty file is NOT a failure
    assert report["ingested"] == 1
    assert report["empty_sources"][0].endswith("empty.txt")


def test_checkpoint_file_created_and_contains_hashes(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "a.txt", b"The Eiffel Tower is in Paris and stands 330 metres tall.")
    _write(docs / "b.txt", b"Photosynthesis converts sunlight into chemical energy in plants.")
    ckpt = tmp_path / "ckpt"

    main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "text"])

    checkpoint = ckpt / "ingested_hashes.json"
    assert checkpoint.exists()
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert isinstance(stored, list)
    assert len(stored) == 2
    # Content is SHA-256 hex (64 lowercase hex chars) as produced by the hasher.
    assert all(len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in stored)


def test_checkpoint_roundtrip_and_atomic_save(tmp_path) -> None:
    path = tmp_path / "sub" / "ingested_hashes.json"
    hashes = {"aaa", "bbb", "ccc"}

    _save_checkpoint(path, hashes)
    assert path.exists()
    assert _load_checkpoint(path) == hashes


def test_load_missing_checkpoint_returns_empty(tmp_path) -> None:
    assert _load_checkpoint(tmp_path / "nope.json") == set()


def test_corrupt_checkpoint_self_repairs_to_empty(tmp_path) -> None:
    path = tmp_path / "ingested_hashes.json"
    path.write_text("{ this is not valid json array", encoding="utf-8")

    # Mirrors FileCheckpointStore: unreadable JSON degrades to an empty set and
    # the bad file is quarantined instead of crashing the run.
    assert _load_checkpoint(path) == set()
    assert not path.exists()
    assert (tmp_path / "ingested_hashes.json.corrupt").exists()


def test_no_supported_files_returns_two(tmp_path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs / "ignore.bin", b"unsupported")  # not in SUPPORTED
    ckpt = tmp_path / "ckpt"

    exit_code = main(["--path", str(docs), "--checkpoint-dir", str(ckpt), "--report", "text"])
    out = capsys.readouterr().out

    assert exit_code == 2
    assert "ingested    : 0" in out


def test_report_dict_shape_is_json_serialisable() -> None:
    report = IngestReport(ingested=["x"], empty=["y"], failed=[("z", "Err: boom")], skipped=["w"], total_chunks=5)
    payload = json.dumps(report.as_dict())  # must not raise
    data = json.loads(payload)
    assert data["ingested"] == 1
    assert data["total_chunks"] == 5
    assert data["failed_sources"][0] == {"source": "z", "reason": "Err: boom"}


def test_module_exposes_main() -> None:
    # `python -m app.rag` dispatches here; guard the public entry point exists.
    assert callable(ingest_cli.main)
