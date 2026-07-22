"""CLI entry point for the offline RAG benchmark suite.

Run with::

    python -m app.bench [--repeat N] [--n-vectors N]
                        [--batch-size N] [--sim-latency-ms MS]

Runs every benchmark with a fully offline stack (hash embedder + in-memory
store, no network), prints a markdown table of real numbers, and exits ``0``.
The delay-mock rows are a *simulated* network latency; the table's notes say
so.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.bench import format_markdown, run_benchmarks
from app.core import get_logger

logger = get_logger(__name__)

_DEFAULT_REPEAT = 3
_DEFAULT_N_VECTORS = 500
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_SIM_LATENCY_MS = 1.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the CLI arguments for the benchmark suite.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(prog="app.bench", description="Offline RAG benchmark suite")
    parser.add_argument(
        "--repeat",
        type=int,
        default=_DEFAULT_REPEAT,
        help="Timed repetitions to average over (default: %(default)s).",
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=_DEFAULT_N_VECTORS,
        help="Vectors pre-loaded for the retrieve benchmark (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size for the embedding benchmarks (default: %(default)s).",
    )
    parser.add_argument(
        "--sim-latency-ms",
        type=float,
        default=_DEFAULT_SIM_LATENCY_MS,
        help="Simulated per-call latency (ms) for the delay-mock (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmarks, print the markdown table, exit 0.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` always - the suite is a reporting tool, not a gate.
    """
    args = _parse_args(argv)
    rows = asyncio.run(
        run_benchmarks(
            repeat=args.repeat,
            n_vectors=args.n_vectors,
            batch_size=args.batch_size,
            sim_latency_ms=args.sim_latency_ms,
        )
    )
    print(format_markdown(rows, repeat=args.repeat))
    logger.info(
        "Bench done: repeat=%d n_vectors=%d batch_size=%d sim_latency_ms=%.1f",
        args.repeat,
        args.n_vectors,
        args.batch_size,
        args.sim_latency_ms,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
