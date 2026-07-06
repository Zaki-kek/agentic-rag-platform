"""Pure, dependency-free metric functions for RAG answer evaluation.

Each metric returns a float in ``[0, 1]`` and never touches the network or
disk. They are deliberately simple, ragas-style proxies (keyword recall,
grounding/faithfulness, numeric preservation) rather than learned scorers, so
they run offline with no API keys.
"""

from __future__ import annotations

import re

# Words shorter than this are treated as stop-words for the grounding proxy,
# which keeps articles and prepositions ("a", "of", "the") from inflating it.
_MIN_CONTENT_WORD_LEN = 3

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _content_words(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens of meaningful length.

    Args:
        text: Arbitrary input text.

    Returns:
        Tokens at least ``_MIN_CONTENT_WORD_LEN`` characters long, lowercased.
    """
    return [tok for tok in _WORD_RE.findall(text.lower()) if len(tok) >= _MIN_CONTENT_WORD_LEN]


def _format_number(value: float) -> str:
    """Render a number the way it would naturally appear in prose.

    Integers lose their trailing ``.0`` so that ``42.0`` matches ``"42"``.

    Args:
        value: The reference number.

    Returns:
        A compact string form of the number.
    """
    if value == int(value):
        return str(int(value))
    return repr(value)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive).

    Args:
        answer: The model's answer text.
        expected_keywords: Keywords a correct answer should mention.

    Returns:
        Fraction of keywords found, in ``[0, 1]``. Returns ``1.0`` when no
        keywords are required.
    """
    if not expected_keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return hits / len(expected_keywords)


def grounding(answer: str, contexts: list[str]) -> float:
    """Fraction of the answer's content words that appear in the contexts.

    A faithfulness proxy: an answer is "grounded" when most of its substantive
    words are traceable to the retrieved contexts rather than invented.

    Args:
        answer: The model's answer text.
        contexts: Retrieved context strings the answer should be based on.

    Returns:
        Fraction of grounded content words, in ``[0, 1]``. Returns ``1.0`` when
        the answer has no content words.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 1.0
    context_vocab = set(_content_words(" ".join(contexts)))
    grounded = sum(1 for word in answer_words if word in context_vocab)
    return grounded / len(answer_words)


def numbers_preserved(answer: str, reference_numbers: list[float]) -> float:
    """Fraction of reference numbers that appear verbatim in the answer.

    Numbers are matched against the answer's own tokenised numbers so that a
    reference ``42`` is not spuriously matched inside ``"426"``.

    Args:
        answer: The model's answer text.
        reference_numbers: Numbers a faithful answer must reproduce.

    Returns:
        Fraction of preserved numbers, in ``[0, 1]``. Returns ``1.0`` when no
        numbers are required.
    """
    if not reference_numbers:
        return 1.0
    answer_numbers = {_normalize_number(tok) for tok in _NUMBER_RE.findall(answer)}
    preserved = sum(1 for value in reference_numbers if _format_number(value) in answer_numbers)
    return preserved / len(reference_numbers)


def _normalize_number(token: str) -> str:
    """Normalize a numeric token so ``"42.0"`` and ``"42"`` compare equal.

    Args:
        token: A numeric substring extracted from text.

    Returns:
        The canonical string form produced by :func:`_format_number`.
    """
    return _format_number(float(token))
