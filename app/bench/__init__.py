"""Offline micro-benchmarks for the RAG stack (``python -m app.bench``).

The suite measures four things with :func:`time.perf_counter` and zero network
calls, and prints the results as a markdown table:

1. **Embed throughput** on :class:`app.rag.embed.HashEmbedder`, batched via
   :func:`app.rag.embed.embed_in_batches` versus one call per text. Both numbers
   are reported as fact. On the offline hash embedder batching does *not* speed
   things up - it is the same Python loop over the same tokens - so this is an
   honest baseline, not a win. The batching win shows up only on a networked
   provider with per-request overhead.
2. **Embed wall-clock on a delay-mock** (:class:`DelayEmbedder`, one
   ``asyncio.sleep(latency_s)`` per ``embed`` call). Batching 100 texts at
   ``batch_size=32`` makes **4** "network" calls instead of **100**, so the
   batched wall-clock is far lower. This is the structural effect batching
   exists for, isolated from real vectors.
3. **Retrieve latency** (p50/p95) over a store pre-loaded with ``n_vectors``
   vectors, using :func:`numpy.percentile`.
4. **Ingest latency** on a synthetic document.

Every number comes from a real run; nothing here is hard-coded. The delay-mock
result is a *simulated* network latency, and the report and README say so.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

import numpy as np

from app.core import get_logger
from app.rag.embed import Embedder, HashEmbedder, embed_in_batches
from app.rag.pipeline import RagPipeline
from app.rag.store import InMemoryVectorStore

logger = get_logger(__name__)

# Fixed shapes for the synthetic workload. The corpus/query text is neutral
# filler; only its size and token spread matter for timing.
_EMBED_DIM = 256
_THROUGHPUT_TEXTS = 100
_RETRIEVE_QUERIES = 30
_RETRIEVE_K = 5
_INGEST_PARAGRAPH = "The Eiffel Tower is in Paris and stands 330 metres tall. "
_INGEST_REPEAT = 40


class DelayEmbedder:
    """Wrap an embedder and add a fixed per-call ``asyncio.sleep`` latency.

    This stands in for a networked embedding provider whose cost is dominated by
    per-request round-trips rather than per-text compute: every ``embed`` call -
    regardless of how many texts it carries - pays ``latency_s`` once. Batching
    therefore reduces total wall-clock in proportion to the number of calls
    saved. The wrapped embedder still produces the real vectors, so the mock is
    only about *timing*, and a public ``calls`` counter records how many
    provider round-trips happened.

    Attributes:
        dim: Output dimension, inherited from the wrapped embedder.
        calls: Number of ``embed`` calls made (i.e. simulated round-trips).
    """

    def __init__(self, base: Embedder, latency_s: float) -> None:
        self.dim = base.dim
        self._base = base
        self._latency_s = latency_s
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Sleep once for the simulated round-trip, then delegate to the base.

        Args:
            texts: The texts to embed in this (single) provider call.

        Returns:
            One embedding vector per input text, from the wrapped embedder.
        """
        self.calls += 1
        await asyncio.sleep(self._latency_s)
        return await self._base.embed(texts)


@dataclass(frozen=True)
class BenchRow:
    """One measured benchmark line for the markdown report.

    Attributes:
        name: Human-readable benchmark name (first table column).
        metric: The measured quantity and its unit (second column).
        value: The measured value, already rounded for display (third column).
        note: A short honesty/context note (fourth column).
    """

    name: str
    metric: str
    value: str
    note: str


def _synthetic_texts(count: int) -> list[str]:
    """Return ``count`` short, neutral, distinct texts for embedding.

    Args:
        count: Number of texts to generate.

    Returns:
        A list of distinct filler strings (distinct so a real provider could
        not collapse them, though the hash embedder does not care).
    """
    return [f"sample text number {i} about various neutral topics" for i in range(count)]


async def _time_async(fn: object, repeat: int) -> float:
    """Return the mean wall-clock seconds of awaiting ``fn`` ``repeat`` times.

    Args:
        fn: A zero-argument coroutine function to time.
        repeat: How many times to run it (mean is returned; ``repeat`` is
            clamped to at least 1).

    Returns:
        The mean elapsed seconds across the runs.
    """
    runs = max(1, repeat)
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        await fn()  # type: ignore[operator]
        samples.append(time.perf_counter() - start)
    return statistics.mean(samples)


async def bench_embed_throughput(batch_size: int, repeat: int) -> list[BenchRow]:
    """Measure hash-embedder throughput: batched versus one call per text.

    Args:
        batch_size: Batch size for :func:`embed_in_batches`.
        repeat: Number of timed repetitions to average over.

    Returns:
        Two :class:`BenchRow` rows (batched and per-text), reported as fact.
        On this offline embedder the two are within noise of each other - that
        is the honest point.
    """
    embedder = HashEmbedder(dim=_EMBED_DIM)
    texts = _synthetic_texts(_THROUGHPUT_TEXTS)

    async def _batched() -> None:
        await embed_in_batches(embedder, texts, batch_size)

    async def _one_by_one() -> None:
        for text in texts:
            await embedder.embed([text])

    batched_s = await _time_async(_batched, repeat)
    single_s = await _time_async(_one_by_one, repeat)
    n = len(texts)
    return [
        BenchRow(
            name="embed throughput (hash, batched)",
            metric="texts/sec",
            value=f"{n / batched_s:,.0f}",
            note=f"batch_size={batch_size}; offline hash - batching does not speed this up",
        ),
        BenchRow(
            name="embed throughput (hash, per-text)",
            metric="texts/sec",
            value=f"{n / single_s:,.0f}",
            note="same Python loop; real win is on a networked provider",
        ),
    ]


async def bench_embed_delay(batch_size: int, latency_s: float, repeat: int) -> list[BenchRow]:
    """Measure wall-clock on a delay-mock: batched (few calls) versus per-text.

    The mock sleeps once per ``embed`` call, so batching 100 texts at
    ``batch_size`` collapses 100 simulated round-trips into
    ``ceil(100 / batch_size)`` and the batched wall-clock drops accordingly.
    This is the structural effect batching exists for.

    Args:
        batch_size: Batch size for :func:`embed_in_batches`.
        latency_s: Simulated per-call latency in seconds.
        repeat: Number of timed repetitions to average over.

    Returns:
        Two :class:`BenchRow` rows (batched and per-text) plus their call counts
        folded into the notes, on a *simulated* network latency.
    """
    embedder = HashEmbedder(dim=_EMBED_DIM)
    texts = _synthetic_texts(_THROUGHPUT_TEXTS)
    batched_calls = -(-len(texts) // batch_size)  # ceil division
    single_calls = len(texts)

    async def _batched() -> None:
        mock = DelayEmbedder(embedder, latency_s)
        await embed_in_batches(mock, texts, batch_size)

    async def _one_by_one() -> None:
        mock = DelayEmbedder(embedder, latency_s)
        for text in texts:
            await mock.embed([text])

    batched_s = await _time_async(_batched, repeat)
    single_s = await _time_async(_one_by_one, repeat)
    latency_ms = latency_s * 1000.0
    return [
        BenchRow(
            name="embed wall-clock (delay-mock, batched)",
            metric="ms (100 texts)",
            value=f"{batched_s * 1000:.1f}",
            note=f"{batched_calls} simulated calls @ {latency_ms:.1f}ms each",
        ),
        BenchRow(
            name="embed wall-clock (delay-mock, per-text)",
            metric="ms (100 texts)",
            value=f"{single_s * 1000:.1f}",
            note=f"{single_calls} simulated calls @ {latency_ms:.1f}ms each",
        ),
    ]


async def bench_retrieve_latency(n_vectors: int, repeat: int) -> list[BenchRow]:
    """Measure retrieve p50/p95 latency over ``n_vectors`` pre-loaded vectors.

    Args:
        n_vectors: Number of vectors to pre-load into the in-memory store.
        repeat: Number of query repetitions per probe (queries are also varied
            across the fixed probe count to spread the percentile sample).

    Returns:
        Two :class:`BenchRow` rows: p50 and p95 retrieve latency in ms.
    """
    embedder = HashEmbedder(dim=_EMBED_DIM)
    store = InMemoryVectorStore()
    await store.init()
    docs = [f"document {i} content words alpha beta gamma delta {i}" for i in range(n_vectors)]
    vectors = await embedder.embed(docs)
    for i, (doc, vector) in enumerate(zip(docs, vectors, strict=True)):
        await store.add(f"doc{i}", [doc], [vector])

    pipeline = RagPipeline(embedder, store)
    probes = max(_RETRIEVE_QUERIES, repeat)
    latencies_ms: list[float] = []
    for i in range(probes):
        query = f"query about alpha beta gamma {i}"
        start = time.perf_counter()
        await pipeline.retrieve(query, _RETRIEVE_K)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return [
        BenchRow(
            name="retrieve latency p50",
            metric="ms",
            value=f"{p50:.3f}",
            note=f"{n_vectors} vectors, top-{_RETRIEVE_K}, in-memory cosine",
        ),
        BenchRow(
            name="retrieve latency p95",
            metric="ms",
            value=f"{p95:.3f}",
            note=f"{n_vectors} vectors, top-{_RETRIEVE_K}, in-memory cosine",
        ),
    ]


async def bench_ingest_latency(repeat: int) -> list[BenchRow]:
    """Measure mean ingest latency on a synthetic multi-paragraph document.

    A fresh pipeline is used per run so no query/dedup state carries over.

    Args:
        repeat: Number of ingest repetitions to average over.

    Returns:
        One :class:`BenchRow` row: mean ingest latency in ms.
    """
    data = (_INGEST_PARAGRAPH * _INGEST_REPEAT).encode("utf-8")
    runs = max(1, repeat)
    samples_ms: list[float] = []
    for i in range(runs):
        pipeline = RagPipeline(HashEmbedder(dim=_EMBED_DIM), InMemoryVectorStore())
        start = time.perf_counter()
        await pipeline.ingest(f"synthetic-{i}.txt", data)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return [
        BenchRow(
            name="ingest latency (synthetic doc)",
            metric="ms",
            value=f"{statistics.mean(samples_ms):.3f}",
            note=f"{len(data)} bytes -> extract+chunk+embed+store",
        ),
    ]


async def run_benchmarks(
    *,
    repeat: int,
    n_vectors: int,
    batch_size: int,
    sim_latency_ms: float,
) -> list[BenchRow]:
    """Run every benchmark and return the combined rows in report order.

    Args:
        repeat: Timed repetitions to average over.
        n_vectors: Vectors pre-loaded for the retrieve benchmark.
        batch_size: Batch size for the embedding benchmarks.
        sim_latency_ms: Simulated per-call latency (ms) for the delay-mock.

    Returns:
        All :class:`BenchRow` rows, ready to render as a markdown table.
    """
    latency_s = sim_latency_ms / 1000.0
    rows: list[BenchRow] = []
    rows += await bench_embed_throughput(batch_size, repeat)
    rows += await bench_embed_delay(batch_size, latency_s, repeat)
    rows += await bench_retrieve_latency(n_vectors, repeat)
    rows += await bench_ingest_latency(repeat)
    return rows


def format_markdown(rows: list[BenchRow], *, repeat: int) -> str:
    """Render benchmark rows as a markdown table with a heading.

    Args:
        rows: The measured rows.
        repeat: Repetition count, shown in the heading for reproducibility.

    Returns:
        A markdown string: a heading followed by a four-column table.
    """
    lines = [
        f"## RAG bench (offline, perf_counter, repeat={repeat})",
        "",
        "| benchmark | metric | value | note |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.name} | {row.metric} | {row.value} | {row.note} |")
    return "\n".join(lines)


__all__ = [
    "BenchRow",
    "DelayEmbedder",
    "bench_embed_delay",
    "bench_embed_throughput",
    "bench_ingest_latency",
    "bench_retrieve_latency",
    "format_markdown",
    "run_benchmarks",
]
