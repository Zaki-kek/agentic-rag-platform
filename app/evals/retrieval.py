"""Pure, offline retrieval-quality metrics for the RAG evaluation harness.

These score a *ranking* of retrieved document ids against a labelled set of
relevant ids. Like the answer metrics they touch neither disk nor network and
return floats in ``[0, 1]``, so they run offline with no API keys.
"""

from __future__ import annotations


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Whether any relevant id appears in the top ``k`` retrieved ids.

    A binary "did we surface at least one right document early" signal.

    Args:
        retrieved_ids: Document ids in retrieval rank order (best first).
        relevant_ids: Ids considered relevant for the query.
        k: Number of leading retrieved ids to inspect. Values ``<= 0`` inspect
            nothing; values larger than the list simply use the whole list.

    Returns:
        ``1.0`` if at least one relevant id is within the top ``k`` retrieved
        ids, otherwise ``0.0``. Returns ``0.0`` when no relevant ids are given.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean reciprocal rank of the first relevant retrieved id.

    Since a single query has one ranking, this is the reciprocal rank: ``1/r``
    where ``r`` is the 1-based position of the first relevant id.

    Args:
        retrieved_ids: Document ids in retrieval rank order (best first).
        relevant_ids: Ids considered relevant for the query.

    Returns:
        The reciprocal of the first relevant id's rank, in ``(0, 1]``, or
        ``0.0`` when no relevant id is retrieved (or none is labelled).
    """
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0
