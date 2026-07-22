"""CLI entry point for the offline RAG evaluation harness.

Run with::

    python -m app.evals [--golden PATH] [--threshold FLOAT]

It loads the golden set, evaluates it against the deterministic echo provider
(no network, no API keys), prints a pass-rate and per-metric report, and exits
``0`` when ``pass_rate`` meets the threshold, otherwise ``1``. This makes the
harness usable as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import get_logger
from app.evals.models import EvalReport, GoldenCase
from app.evals.runner import EvalRunner, load_golden_set
from app.llm.echo import EchoProvider

logger = get_logger(__name__)

# Golden set bundled with the package; used unless --golden overrides it.
_DEFAULT_GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"

_DEFAULT_PASS_THRESHOLD = 0.5


async def _echo_answer_fn(question: str) -> tuple[str, list[str]]:
    """Answer a question with the offline echo provider.

    The echo provider synthesises a grounded-looking answer from the question
    itself; the question is also returned as the single "context" so grounding
    stays fully offline and deterministic.

    Args:
        question: The golden-case question.

    Returns:
        A ``(answer, contexts)`` pair.
    """
    provider = EchoProvider()
    context = f"Context: {question}"
    answer = await provider.generate(
        [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
    )
    return answer, [context]


def _format_report(report: EvalReport, threshold: float) -> str:
    """Render an :class:`EvalReport` as a compact text block.

    Args:
        report: The aggregate evaluation report.
        threshold: The pass-rate threshold used for the exit decision.

    Returns:
        A human-readable multi-line report.
    """
    lines = [
        "=== RAG eval report (echo provider, offline) ===",
        f"cases       : {len(report.cases)}",
        f"pass_rate   : {report.pass_rate:.3f} (threshold {threshold:.3f})",
        "per-metric averages:",
    ]
    for name, value in sorted(report.averages.items()):
        lines.append(f"  {name:<18}: {value:.3f}")
    return "\n".join(lines)


async def _run(cases: list[GoldenCase]) -> EvalReport:
    """Evaluate cases with the echo answer function.

    Args:
        cases: The golden cases to evaluate.

    Returns:
        The aggregate report.
    """
    runner = EvalRunner(_echo_answer_fn)
    return await runner.run(cases)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the eval, print the report and return an exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if ``pass_rate >= threshold``, otherwise ``1``.
    """
    import asyncio

    parser = argparse.ArgumentParser(prog="app.evals", description="Offline RAG eval harness")
    parser.add_argument(
        "--golden",
        type=Path,
        default=_DEFAULT_GOLDEN_SET,
        help="Path to the golden-set JSON file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_PASS_THRESHOLD,
        help="Minimum pass_rate required for a zero exit code.",
    )
    args = parser.parse_args(argv)

    cases = load_golden_set(args.golden)
    report = asyncio.run(_run(cases))

    print(_format_report(report, args.threshold))

    passed = report.pass_rate >= args.threshold
    logger.info(
        "Eval gate %s (pass_rate=%.3f, threshold=%.3f)",
        "PASSED" if passed else "FAILED",
        report.pass_rate,
        args.threshold,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
