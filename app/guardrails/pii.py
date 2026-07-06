"""Regex-based PII detection and redaction (offline, no external services)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PiiMatch:
    """A single piece of detected PII and its location in the source text.

    Attributes:
        kind: The category of PII, one of ``email``, ``phone``, ``card`` or ``ipv4``.
        value: The exact matched substring.
        start: Inclusive start offset of the match in the source text.
        end: Exclusive end offset of the match in the source text.
    """

    kind: str
    value: str
    start: int
    end: int


# Email: standard local@domain.tld shape, kept conservative to avoid trailing punctuation.
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)

# IPv4: four dotted octets, each 0-255, with word boundaries so it does not glue to text.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(
    rf"\b{_OCTET}\.{_OCTET}\.{_OCTET}\.{_OCTET}\b",
)

# Phone numbers: optional leading + or 8, then 10-14 digits split by spaces/dashes/parens.
# Require at least one separator OR a leading +/8 country marker so plain integers are skipped.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s\-]?|8[\s\-]?)?"
    r"(?:\(\d{2,4}\)[\s\-]?)?"
    r"\d{2,4}(?:[\s\-]\d{2,4}){2,4}"
    r"(?![\w.])",
)

# Credit-card-like: 13-16 digits, either contiguous or in 4-digit groups separated by space/dash.
_CARD = re.compile(
    r"(?<![\w.])(?:\d[ \-]?){12,18}\d(?![\w.])",
)


def _card_matches(text: str) -> list[PiiMatch]:
    """Find credit-card-like runs whose digit count is between 13 and 16."""
    matches: list[PiiMatch] = []
    for m in _CARD.finditer(text):
        value = m.group(0)
        digit_count = sum(ch.isdigit() for ch in value)
        if 13 <= digit_count <= 16:
            matches.append(PiiMatch("card", value, m.start(), m.end()))
    return matches


def _phone_matches(text: str, taken: list[tuple[int, int]]) -> list[PiiMatch]:
    """Find phone-like spans, skipping anything overlapping an already-claimed span."""
    matches: list[PiiMatch] = []
    for m in _PHONE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        value = m.group(0)
        digit_count = sum(ch.isdigit() for ch in value)
        if not 7 <= digit_count <= 15:
            continue
        # Require either a +/8 country marker or an explicit separator to avoid plain numbers.
        if "+" not in value and not value.lstrip().startswith("8") and not re.search(r"[\s\-()]", value):
            continue
        matches.append(PiiMatch("phone", value, m.start(), m.end()))
    return matches


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Return True if [start, end) intersects any span in ``spans``."""
    return any(start < s_end and s_start < end for s_start, s_end in spans)


class PiiRedactor:
    """Detects and masks common PII patterns in free text.

    Detection is purely regex-based and runs fully offline. Patterns are tuned to
    avoid over-matching plain integers (such as ``42`` or a year like ``2026``).
    More specific categories (``email``, ``ipv4``, ``card``) are matched before
    the looser ``phone`` pattern so overlapping spans are not double-claimed.
    """

    def find(self, text: str) -> list[PiiMatch]:
        """Return all PII matches in ``text`` ordered by start offset.

        Args:
            text: The text to scan.

        Returns:
            A list of :class:`PiiMatch`, sorted by their start offset. Overlapping
            candidates are resolved in favor of the more specific category.
        """
        matches: list[PiiMatch] = []
        for kind, pattern in (("email", _EMAIL), ("ipv4", _IPV4)):
            for m in pattern.finditer(text):
                matches.append(PiiMatch(kind, m.group(0), m.start(), m.end()))

        matches.extend(_card_matches(text))

        taken = [(pm.start, pm.end) for pm in matches]
        matches.extend(_phone_matches(text, taken))

        matches.sort(key=lambda pm: pm.start)
        logger.debug("PII find: %d match(es)", len(matches))
        return matches

    def redact(self, text: str, mask: str = "[REDACTED]") -> str:
        """Replace every detected PII span with ``mask``.

        Args:
            text: The text to redact.
            mask: The replacement string substituted for each detected span.

        Returns:
            The redacted text with all PII spans replaced by ``mask``.
        """
        matches = self.find(text)
        if not matches:
            return text
        # Drop any span overlapping an earlier (already-emitted) one, then splice right-to-left.
        kept: list[PiiMatch] = []
        last_end = -1
        for pm in matches:
            if pm.start >= last_end:
                kept.append(pm)
                last_end = pm.end
        result = text
        for pm in reversed(kept):
            result = result[: pm.start] + mask + result[pm.end :]
        return result


_DEFAULT = PiiRedactor()


def redact_pii(text: str, mask: str = "[REDACTED]") -> str:
    """Convenience wrapper that redacts ``text`` with a shared default redactor.

    Args:
        text: The text to redact.
        mask: The replacement string substituted for each detected span.

    Returns:
        The redacted text.
    """
    return _DEFAULT.redact(text, mask=mask)
