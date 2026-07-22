"""CLI entry point for the offline RAG evaluation harness.

Run with::

    python -m app.evals [--golden PATH] [--threshold FLOAT]
    python -m app.evals --judge        # ensemble judge: ECE + AUC separation
    python -m app.evals --retrieval    # retrieval quality: hit@k + MRR

Without a mode flag it loads the golden set, evaluates it against the
deterministic echo provider (no network, no API keys), prints a pass-rate and
per-metric report, and exits ``0`` when ``pass_rate`` meets the threshold,
otherwise ``1``. This makes the harness usable as a CI gate. The ``--judge`` and
``--retrieval`` flags are additive diagnostics that print real numbers and
always exit ``0``; they leave the default report untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.core import get_logger
from app.evals.calibration import expected_calibration_error
from app.evals.composite import composite_faithfulness, separation
from app.evals.judge import EnsembleJudge, JudgedCase, load_judged_set
from app.evals.models import EvalReport, GoldenCase
from app.evals.retrieval import hit_at_k, mrr
from app.evals.runner import EvalRunner, load_golden_set
from app.llm.echo import EchoProvider
from app.rag.embed import HashEmbedder
from app.rag.store import InMemoryVectorStore

logger = get_logger(__name__)

# Golden set bundled with the package; used unless --golden overrides it.
_DEFAULT_GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"

_DEFAULT_PASS_THRESHOLD = 0.5

# Retrieval diagnostic settings.
_RETRIEVAL_K = 3
_RETRIEVAL_SEARCH_DEPTH = 5
_EMBED_DIM = 256


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


def _judge_metric_columns(
    judge: EnsembleJudge, cases: list[JudgedCase]
) -> dict[str, tuple[list[float], list[float]]]:
    """Split per-metric and composite scores into good/bad columns.

    Args:
        judge: The ensemble judge providing the raw metric values.
        cases: The judged cases to score.

    Returns:
        A mapping from column name (``"grounding"``, ``"keyword"``,
        ``"numbers"``, ``"composite"``) to a ``(good_scores, bad_scores)``
        pair, where the split follows each case's gold label.
    """
    names = ["grounding", "keyword", "numbers", "composite"]
    columns: dict[str, tuple[list[float], list[float]]] = {n: ([], []) for n in names}
    for case in cases:
        values = judge.metric_values(case)
        composite = composite_faithfulness(
            values["grounding"], values["keyword"], values["numbers"]
        )
        scored = {**values, "composite": composite}
        bucket = 0 if case.label == "good" else 1
        for name in names:
            columns[name][bucket].append(scored[name])
    return columns


def _format_judge_report(cases: list[JudgedCase]) -> str:
    """Run the ensemble judge and render its calibration/separation report.

    Args:
        cases: The judged cases (hand-written good/bad pairs).

    Returns:
        A human-readable multi-line report with ECE, per-case confidence, and
        the AUC separation of the composite score versus each single metric.
    """
    judge = EnsembleJudge()
    results = [(case, judge.judge(case)) for case in cases]

    confidences = [res.confidence for _, res in results]
    # A judgement is "correct" when its predicted label matches the gold label.
    correct = [1.0 if res.predicted_label == case.label else 0.0 for case, res in results]
    ece = expected_calibration_error(confidences, correct)

    columns = _judge_metric_columns(judge, cases)
    separations = {name: separation(good, bad) for name, (good, bad) in columns.items()}

    lines = [
        "=== Ensemble judge report (offline, deterministic) ===",
        f"cases            : {len(cases)}",
        f"ECE (confidence) : {ece:.3f}",
        "per-case confidence:",
    ]
    for case, res in results:
        flag = "ok " if res.predicted_label == case.label else "MISS"
        lines.append(
            f"  [{flag}] {case.label:<4} conf={res.confidence:.3f} score={res.score:.3f}"
            f"  {case.question[:48]}"
        )
    lines.append("AUC separation good>bad (higher = cleaner signal):")
    for name in ["composite", "grounding", "keyword", "numbers"]:
        lines.append(f"  {name:<10}: {separations[name]:.3f}")
    best_single = max(separations[n] for n in ["grounding", "keyword", "numbers"])
    lines.append(
        f"composite ({separations['composite']:.3f}) vs best single "
        f"({best_single:.3f}): "
        f"{'>=' if separations['composite'] >= best_single else '<'} best single metric"
    )
    return "\n".join(lines)


def _build_retrieval_corpus(cases: list[GoldenCase]) -> dict[str, str]:
    """Build a toy doc-id -> text corpus from golden ``relevant_doc_ids``.

    Each relevant document id is given the text of the first golden question
    that references it, yielding a small deterministic corpus the hash embedder
    can retrieve against.

    Args:
        cases: The golden cases carrying ``relevant_doc_ids``.

    Returns:
        A mapping from document id to a representative text, in first-seen
        order.
    """
    corpus: dict[str, str] = {}
    for case in cases:
        for doc_id in case.relevant_doc_ids:
            corpus.setdefault(doc_id, case.question)
    return corpus


async def _run_retrieval(cases: list[GoldenCase]) -> tuple[float, float, int]:
    """Score retrieval quality on the toy corpus with hit@k and MRR.

    Args:
        cases: The golden cases (only those with ``relevant_doc_ids`` count).

    Returns:
        A ``(mean_hit_at_k, mean_mrr, num_queries)`` triple.
    """
    corpus = _build_retrieval_corpus(cases)
    embedder = HashEmbedder(dim=_EMBED_DIM)
    store = InMemoryVectorStore()
    await store.init()

    doc_ids = list(corpus)
    texts = [corpus[doc_id] for doc_id in doc_ids]
    embeddings = await embedder.embed(texts)
    for doc_id, text, embedding in zip(doc_ids, texts, embeddings, strict=True):
        await store.add(doc_id, [text], [embedding])

    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        if not case.relevant_doc_ids:
            continue
        query_vec = (await embedder.embed([case.question]))[0]
        results = await store.search(query_vec, _RETRIEVAL_SEARCH_DEPTH)
        ranked_ids = [hit.document for hit in results]
        hits.append(hit_at_k(ranked_ids, case.relevant_doc_ids, _RETRIEVAL_K))
        reciprocal_ranks.append(mrr(ranked_ids, case.relevant_doc_ids))

    if not hits:
        return 0.0, 0.0, 0
    mean_hit = sum(hits) / len(hits)
    mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mean_hit, mean_mrr, len(hits)


def _format_retrieval_report(mean_hit: float, mean_mrr: float, num_queries: int) -> str:
    """Render the retrieval-quality report.

    Args:
        mean_hit: Mean hit@k over the labelled queries.
        mean_mrr: Mean reciprocal rank over the labelled queries.
        num_queries: Number of queries scored.

    Returns:
        A human-readable multi-line report.
    """
    return "\n".join(
        [
            "=== Retrieval quality report (HashEmbedder + InMemoryVectorStore) ===",
            f"queries     : {num_queries}",
            f"hit@{_RETRIEVAL_K}      : {mean_hit:.3f}",
            f"MRR         : {mean_mrr:.3f}",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the CLI arguments for the eval harness.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed argument namespace.
    """
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
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run the ensemble judge: print ECE and AUC separation, then exit 0.",
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Run retrieval metrics (hit@k, MRR) on the golden corpus, then exit 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the requested eval, print a report and exit code.

    With no mode flag this is the original CI gate (exit ``0`` iff
    ``pass_rate >= threshold``). ``--judge`` and ``--retrieval`` are additive
    diagnostics that print real numbers and always return ``0``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success (always for the diagnostics), otherwise ``1``.
    """
    args = _parse_args(argv)

    if args.judge:
        cases = load_judged_set()
        print(_format_judge_report(cases))
        return 0

    if args.retrieval:
        golden = load_golden_set(args.golden)
        mean_hit, mean_mrr, num_queries = asyncio.run(_run_retrieval(golden))
        print(_format_retrieval_report(mean_hit, mean_mrr, num_queries))
        return 0

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
