"""Tests for the offline RAG benchmark suite (``python -m app.bench``).

These pin what the benchmark suite must guarantee: it runs entirely offline (no
network), every measured value is positive, and the *structural* batching win on
the delay-mock holds - batching 100 texts into 4 simulated round-trips is faster
than 100 round-trips. That invariant compares ``4`` sleeps against ``100``, so
it is robust rather than a flaky wall-clock threshold. The suite deliberately
does **not** claim a batching win on the offline hash embedder (there is none -
it is the same Python loop), so no test asserts hash-batch superiority. The
markdown report is checked for its expected headers.
"""

from __future__ import annotations

from app.bench import (
    BenchRow,
    DelayEmbedder,
    bench_embed_delay,
    format_markdown,
    run_benchmarks,
)
from app.rag.embed import HashEmbedder


def _value(rows: list[BenchRow], name: str) -> float:
    """Return the numeric value of the row whose name contains ``name``.

    Args:
        rows: Benchmark rows to search.
        name: A substring identifying the target row.

    Returns:
        The row's value parsed as a float (thousands separators stripped).
    """
    for row in rows:
        if name in row.name:
            return float(row.value.replace(",", ""))
    raise AssertionError(f"no bench row matching {name!r}")


async def test_all_measurements_are_positive() -> None:
    # A fast, offline configuration: few vectors, one repetition. Every measured
    # number the suite reports must be strictly positive (a zero/negative timing
    # would signal a broken measurement).
    rows = await run_benchmarks(repeat=1, n_vectors=50, batch_size=32, sim_latency_ms=1.0)
    assert rows, "benchmark suite produced no rows"
    for row in rows:
        assert float(row.value.replace(",", "")) > 0.0, f"{row.name} was not positive"


async def test_delay_mock_batching_beats_per_text() -> None:
    # Structural invariant, not a fragile threshold: on the delay-mock, batching
    # 100 texts at batch_size=32 makes 4 simulated round-trips vs 100, so the
    # batched wall-clock must be lower. The gap is 4 sleeps vs 100 sleeps.
    rows = await bench_embed_delay(batch_size=32, latency_s=0.001, repeat=1)
    batched = _value(rows, "batched")
    per_text = _value(rows, "per-text")
    assert batched < per_text
    # The notes record the call counts the invariant rests on.
    batched_note = next(r.note for r in rows if "batched" in r.name)
    per_text_note = next(r.note for r in rows if "per-text" in r.name)
    assert "4 simulated calls" in batched_note
    assert "100 simulated calls" in per_text_note


async def test_delay_mock_counts_provider_calls() -> None:
    # The DelayEmbedder must count exactly one call per embed() invocation, so a
    # single batched call over many texts is one simulated round-trip.
    embedder = DelayEmbedder(HashEmbedder(dim=256), latency_s=0.0)
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    vectors = await embedder.embed(texts)
    assert embedder.calls == 1  # one batched call, not one per text
    assert len(vectors) == len(texts)  # real vectors still returned
    assert embedder.dim == 256


def test_markdown_report_has_expected_headers() -> None:
    rows = [BenchRow(name="demo bench", metric="ms", value="1.2", note="context")]
    report = format_markdown(rows, repeat=3)
    # Heading, the four-column header, its separator, and the data row.
    assert "## RAG bench (offline, perf_counter, repeat=3)" in report
    assert "| benchmark | metric | value | note |" in report
    assert "| --- | --- | --- | --- |" in report
    assert "| demo bench | ms | 1.2 | context |" in report


def test_cli_main_prints_markdown_and_exits_zero(capsys) -> None:
    from app.bench.__main__ import main

    # A tiny run so the test stays fast; the CLI must print the markdown table
    # and return 0 (it is a reporting tool, never a gate).
    exit_code = main(["--repeat", "1", "--n-vectors", "20", "--batch-size", "32", "--sim-latency-ms", "1"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "## RAG bench" in out
    assert "| benchmark | metric | value | note |" in out
    # Both delay-mock rows appear, carrying the 4-vs-100 structural story.
    assert "4 simulated calls" in out
    assert "100 simulated calls" in out
