"""Offline batch-ingest CLI for the RAG pipeline.

Walks a directory (or an explicit file list), ingests every supported document
through :class:`app.rag.pipeline.RagPipeline` with a fully offline stack
(:class:`app.rag.embed.HashEmbedder` + :class:`app.rag.store.InMemoryVectorStore`),
and prints a per-run report. It is built for *batch* ingest, so it must not
abort the whole run because one file is malformed: each file is isolated behind
a broad ``except Exception`` and classified as ``ingested``, ``empty``,
``failed`` or ``skipped``. The distinction between a validly-empty document and
a genuinely broken one is deliberate - an empty ``.txt`` is ``empty``, a
garbage ``.pdf`` that makes the parser raise is ``failed``.

A JSON checkpoint of already-ingested content hashes makes re-runs cheap: a
document whose bytes were ingested on a previous run is reported as ``skipped``
and never re-embedded. The checkpoint reuses the durability pattern of
:class:`app.generation.checkpoint.FileCheckpointStore` - atomic write via a
temporary file plus :func:`os.replace`, and self-repair (quarantine to
``.corrupt`` and start empty) when the file is unreadable JSON - but keeps its
own ``set[str]`` shape rather than importing that ``JobState``-typed class.

Run with::

    python -m app.rag --path DIR [--checkpoint-dir DIR] [--report json|text]
    python -m app.rag file1.txt file2.pdf --report json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.core import get_logger
from app.rag.embed import HashEmbedder
from app.rag.extract import SUPPORTED
from app.rag.metadata import compute_content_hash
from app.rag.pipeline import RagPipeline
from app.rag.store import InMemoryVectorStore

logger = get_logger(__name__)

_DEFAULT_CHECKPOINT_DIR = ".ingest_checkpoints"
_CHECKPOINT_FILENAME = "ingested_hashes.json"


@dataclass
class IngestReport:
    """Aggregate outcome of a batch-ingest run.

    Attributes:
        ingested: Sources that produced at least one chunk.
        empty: Sources that were read successfully but yielded no chunks
            (e.g. an empty text file); these are valid, not failures.
        failed: ``(source, reason)`` pairs for files whose processing raised;
            ``reason`` is ``"ErrorType: message"``.
        skipped: Sources whose content hash was already in the checkpoint.
        total_chunks: Sum of chunk counts across all ``ingested`` sources.
    """

    ingested: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_chunks: int = 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of the run.

        Returns:
            A mapping with the per-category counts, the list of sources in
            each category and ``total_chunks``.
        """
        return {
            "ingested": len(self.ingested),
            "empty": len(self.empty),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "total_chunks": self.total_chunks,
            "ingested_sources": list(self.ingested),
            "empty_sources": list(self.empty),
            "failed_sources": [{"source": s, "reason": r} for s, r in self.failed],
            "skipped_sources": list(self.skipped),
        }


def _checkpoint_path(checkpoint_dir: str | Path) -> Path:
    """Return the checkpoint file path inside ``checkpoint_dir``.

    Args:
        checkpoint_dir: Directory that holds (or will hold) the checkpoint.

    Returns:
        The full path to the JSON checkpoint file.
    """
    return Path(checkpoint_dir) / _CHECKPOINT_FILENAME


def _load_checkpoint(path: str | Path) -> set[str]:
    """Load the set of already-ingested content hashes from ``path``.

    Mirrors :class:`app.generation.checkpoint.FileCheckpointStore` self-repair:
    a missing file yields an empty set, and a file that is not readable as a
    JSON array is quarantined to ``.corrupt`` and treated as empty so a
    damaged checkpoint never aborts a run.

    Args:
        path: Path to the JSON checkpoint file.

    Returns:
        The set of stored content-hash strings (empty if absent or corrupt).
    """
    checkpoint = Path(path)
    if not checkpoint.exists():
        return set()
    try:
        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        corrupt = checkpoint.with_suffix(checkpoint.suffix + ".corrupt")
        os.replace(checkpoint, corrupt)
        logger.warning("Corrupt ingest checkpoint moved to %s; starting empty", corrupt.name)
        return set()
    if not isinstance(raw, list):
        corrupt = checkpoint.with_suffix(checkpoint.suffix + ".corrupt")
        os.replace(checkpoint, corrupt)
        logger.warning("Unexpected ingest checkpoint shape moved to %s; starting empty", corrupt.name)
        return set()
    return {str(item) for item in raw}


def _save_checkpoint(path: str | Path, hashes: set[str]) -> None:
    """Atomically persist the set of ingested content hashes to ``path``.

    Writes to a sibling temporary file and :func:`os.replace` it into place so
    a crash mid-write cannot leave a half-written checkpoint (same guarantee as
    :class:`app.generation.checkpoint.FileCheckpointStore`).

    Args:
        path: Path to the JSON checkpoint file.
        hashes: Content hashes to record (serialised as a sorted JSON array).
    """
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    tmp.write_text(json.dumps(sorted(hashes), indent=2), encoding="utf-8")
    os.replace(tmp, checkpoint)  # atomic on the same filesystem


def _discover_files(path: Path | None, files: list[str]) -> list[Path]:
    """Collect supported document paths from a directory and/or explicit files.

    A directory is walked recursively; explicit files are taken as-is. In both
    cases only paths whose suffix is in :data:`app.rag.extract.SUPPORTED` are
    kept, so unsupported files never reach the parser. Results are sorted for
    deterministic ordering.

    Args:
        path: Optional directory to walk recursively.
        files: Explicit file paths supplied on the command line.

    Returns:
        A sorted, de-duplicated list of supported document paths.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.extend(p for p in path.rglob("*") if p.is_file())
    candidates.extend(Path(f) for f in files)
    supported = {p.resolve() for p in candidates if p.suffix.lower() in SUPPORTED}
    return sorted(supported)


async def _ingest_files(
    paths: list[Path],
    pipeline: RagPipeline,
    seen_hashes: set[str],
) -> IngestReport:
    """Ingest each path in isolation and classify its outcome.

    Every file is processed behind a broad ``except Exception`` so one broken
    document cannot abort the batch. A file whose content hash is already in
    ``seen_hashes`` is ``skipped`` before any read; a successful ingest that
    yields chunks is ``ingested`` (and the hash is recorded), a successful read
    that yields no chunks is ``empty``, and any raised error is ``failed``.

    Args:
        paths: Supported document paths to ingest, in order.
        pipeline: The offline RAG pipeline to ingest through.
        seen_hashes: Content hashes already ingested (mutated in place with
            each newly ingested document so the caller can persist them).

    Returns:
        The aggregate :class:`IngestReport` for this run.
    """
    report = IngestReport()
    for file_path in paths:
        source = str(file_path)
        try:
            data = file_path.read_bytes()
            content_hash = compute_content_hash(data)
            if content_hash in seen_hashes:
                report.skipped.append(source)
                logger.info("Skipping already-ingested %s", source)
                continue
            chunk_count = await pipeline.ingest(source, data, dedup=True, skip_duplicates=True)
            if chunk_count > 0:
                report.ingested.append(source)
                report.total_chunks += chunk_count
                seen_hashes.add(content_hash)
                logger.info("Ingested %s (%d chunks)", source, chunk_count)
            else:
                report.empty.append(source)
                logger.info("Empty (no chunks) %s", source)
        except Exception as exc:  # noqa: BLE001 - per-file isolation: one bad file must not abort the batch
            reason = f"{type(exc).__name__}: {exc}"
            report.failed.append((source, reason))
            logger.warning("Failed to ingest %s: %s", source, reason)
    return report


def _format_report_text(report: IngestReport) -> str:
    """Render an :class:`IngestReport` as a compact text block.

    Args:
        report: The aggregate report to render.

    Returns:
        A human-readable multi-line report with the per-category counts.
    """
    lines = [
        "=== RAG ingest report (hash embedder + memory store, offline) ===",
        f"ingested    : {len(report.ingested)}",
        f"empty       : {len(report.empty)}",
        f"failed      : {len(report.failed)}",
        f"skipped     : {len(report.skipped)}",
        f"total_chunks: {report.total_chunks}",
    ]
    for source, reason in report.failed:
        lines.append(f"  failed: {source} ({reason})")
    return "\n".join(lines)


def _format_report(report: IngestReport, fmt: str) -> str:
    """Render an :class:`IngestReport` in the requested format.

    Args:
        report: The aggregate report to render.
        fmt: ``"json"`` for a JSON object or ``"text"`` for a text block.

    Returns:
        The rendered report string.
    """
    if fmt == "json":
        return json.dumps(report.as_dict(), indent=2)
    return _format_report_text(report)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingest CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(prog="app.rag", description="Offline batch RAG ingest")
    parser.add_argument("files", nargs="*", help="Explicit document paths to ingest.")
    parser.add_argument("--path", type=Path, default=None, help="Directory to walk recursively for documents.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size passed to the pipeline.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(_DEFAULT_CHECKPOINT_DIR),
        help="Directory holding the JSON checkpoint of ingested content hashes.",
    )
    parser.add_argument("--report", choices=("json", "text"), default="text", help="Report output format.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the batch ingest, print the report and return a code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when the run completes (even if some files failed - batch ingest
        is resilient by design), ``2`` when no supported inputs were found.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    paths = _discover_files(args.path, args.files)
    if not paths:
        print(_format_report(IngestReport(), args.report))
        logger.warning("No supported documents found under --path/files")
        return 2

    checkpoint_path = _checkpoint_path(args.checkpoint_dir)
    seen_hashes = _load_checkpoint(checkpoint_path)

    embedder = HashEmbedder()
    store = InMemoryVectorStore()
    pipeline = RagPipeline(embedder, store)
    logger.info("Ingesting %d file(s) with batch_size=%d", len(paths), args.batch_size)

    report = asyncio.run(_ingest_files(paths, pipeline, seen_hashes))
    _save_checkpoint(checkpoint_path, seen_hashes)

    print(_format_report(report, args.report))
    logger.info(
        "Ingest done: ingested=%d empty=%d failed=%d skipped=%d total_chunks=%d",
        len(report.ingested),
        len(report.empty),
        len(report.failed),
        len(report.skipped),
        report.total_chunks,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
