"""Extra neutral quality gates: leftover placeholders, minimum length, references."""

from __future__ import annotations

import re
from typing import Any

from app.core import get_logger
from app.generation.models import GateResult

logger = get_logger(__name__)

# Mustache-style template slots like {{ name }} (allows surrounding whitespace).
_MUSTACHE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
# Bracketed editorial markers, case-insensitive: [TODO], [ TBD ], etc.
_BRACKET_MARKER_RE = re.compile(r"\[\s*(?:TODO|TBD|FIXME|XXX)\s*\]", re.IGNORECASE)
# Bare markers as standalone whole words (uppercase only, to avoid normal prose).
_BARE_MARKER_RE = re.compile(r"\b(?:TBD|XXX|FIXME)\b")
# Citation markers like [1], [42].
_REFERENCE_RE = re.compile(r"\[\d+\]")


def find_placeholders(text: str) -> list[str]:
    """Find leftover template placeholders in ``text``.

    Detects three families of unfinished-template markers:

    * mustache-style slots, e.g. ``{{ name }}``;
    * bracketed editorial markers, e.g. ``[TODO]``, ``[TBD]`` (case-insensitive);
    * bare standalone markers ``TBD``, ``XXX``, ``FIXME`` as whole uppercase words.

    The matching is intentionally precise so ordinary prose (for example the word
    "todo" inside a sentence, or a lowercase "xxx") is not flagged.

    Args:
        text: Candidate text to inspect.

    Returns:
        Placeholders found, in order of appearance, including duplicates.
    """
    if not text:
        return []

    matches: list[tuple[int, str]] = []
    for pattern in (_MUSTACHE_RE, _BRACKET_MARKER_RE, _BARE_MARKER_RE):
        for m in pattern.finditer(text):
            matches.append((m.start(), m.group(0)))

    matches.sort(key=lambda item: item[0])
    return [value for _, value in matches]


def references_present(text: str, min_refs: int = 1) -> bool:
    """Report whether ``text`` carries at least ``min_refs`` citation markers.

    Args:
        text: Text to inspect.
        min_refs: Minimum number of ``[n]`` citation markers required.

    Returns:
        True when the count of citation markers is greater than or equal to
        ``min_refs``.
    """
    if not text:
        return min_refs <= 0
    return len(_REFERENCE_RE.findall(text)) >= min_refs


class PlaceholderGate:
    """Fails when a context field still contains template placeholders."""

    def __init__(self, field: str) -> None:
        """Initialize the gate.

        Args:
            field: Context key whose text value is inspected.
        """
        self.name = f"no_placeholders:{field}"
        self._field = field

    def check(self, context: dict[str, Any]) -> GateResult:
        """Check the field for leftover placeholders.

        Args:
            context: Job context mapping.

        Returns:
            A passing :class:`GateResult` when no placeholders are found, otherwise
            a failing result whose reason lists the offending placeholders.
        """
        text = context.get(self._field) or ""
        found = find_placeholders(str(text))
        if found:
            return GateResult(False, f"field '{self._field}' has placeholders: {found}")
        return GateResult(True)


class MinLengthGate:
    """Fails when a context field is shorter than a minimum character count."""

    def __init__(self, field: str, min_chars: int) -> None:
        """Initialize the gate.

        Args:
            field: Context key whose text value is measured.
            min_chars: Minimum acceptable length in characters.
        """
        self.name = f"min_length:{field}>={min_chars}"
        self._field = field
        self._min_chars = min_chars

    def check(self, context: dict[str, Any]) -> GateResult:
        """Check the field's length against the configured minimum.

        Args:
            context: Job context mapping.

        Returns:
            A passing :class:`GateResult` when the length meets the threshold,
            otherwise a failing result describing the shortfall.
        """
        text = context.get(self._field) or ""
        length = len(str(text))
        if length < self._min_chars:
            return GateResult(
                False,
                f"field '{self._field}' too short: {length} < {self._min_chars} chars",
            )
        return GateResult(True)


class ReferencesGate:
    """Fails when a context field lacks enough ``[n]`` citation markers."""

    def __init__(self, field: str, min_refs: int = 1) -> None:
        """Initialize the gate.

        Args:
            field: Context key whose text value is inspected.
            min_refs: Minimum number of citation markers required.
        """
        self.name = f"references:{field}>={min_refs}"
        self._field = field
        self._min_refs = min_refs

    def check(self, context: dict[str, Any]) -> GateResult:
        """Check the field for enough citation markers.

        Args:
            context: Job context mapping.

        Returns:
            A passing :class:`GateResult` when at least ``min_refs`` markers are
            present, otherwise a failing result.
        """
        text = context.get(self._field) or ""
        if references_present(str(text), self._min_refs):
            return GateResult(True)
        return GateResult(False, f"field '{self._field}' needs >= {self._min_refs} citation marker(s) like [1]")
