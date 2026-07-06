"""Unit tests for guardrails: PII redaction and citation validation."""

from __future__ import annotations

from app.guardrails import PiiMatch, PiiRedactor, redact_pii, validate_citations


def test_finds_and_masks_email() -> None:
    redactor = PiiRedactor()
    text = "Contact me at alice.smith@example.com please."
    kinds = {m.kind for m in redactor.find(text)}
    assert "email" in kinds
    redacted = redactor.redact(text)
    assert "alice.smith@example.com" not in redacted
    assert "[REDACTED]" in redacted


def test_finds_and_masks_phone() -> None:
    redactor = PiiRedactor()
    text = "Call +7 905 123-45-67 after noon."
    matches = [m for m in redactor.find(text) if m.kind in {"phone", "card"}]
    assert matches, "expected a phone-like match"
    redacted = redactor.redact(text)
    assert "123-45-67" not in redacted
    assert "[REDACTED]" in redacted


def test_finds_and_masks_ipv4() -> None:
    redactor = PiiRedactor()
    text = "The server lives at 192.168.0.1 on the LAN."
    assert any(m.kind == "ipv4" and m.value == "192.168.0.1" for m in redactor.find(text))
    redacted = redactor.redact(text)
    assert "192.168.0.1" not in redacted
    assert "[REDACTED]" in redacted


def test_finds_and_masks_card_like_number() -> None:
    redactor = PiiRedactor()
    text = "Card 4111 1111 1111 1111 charged today."
    assert any(m.kind == "card" for m in redactor.find(text))
    redacted = redactor.redact(text)
    assert "4111 1111 1111 1111" not in redacted
    assert "[REDACTED]" in redacted


def test_does_not_redact_small_integer() -> None:
    redactor = PiiRedactor()
    text = "The answer is 42 and nothing else."
    assert redactor.find(text) == []
    assert redactor.redact(text) == text


def test_does_not_redact_year() -> None:
    text = "We expect to finish by 2026 for sure."
    assert redact_pii(text) == text


def test_redact_pii_convenience_masks_email() -> None:
    out = redact_pii("ping bob@corp.io now")
    assert "bob@corp.io" not in out
    assert "[REDACTED]" in out


def test_pii_match_is_frozen_dataclass() -> None:
    match = PiiMatch(kind="email", value="x@y.io", start=0, end=6)
    assert (match.kind, match.value, match.start, match.end) == ("email", "x@y.io", 0, 6)


def test_validate_citations_flags_out_of_range() -> None:
    problems = validate_citations("As shown in [5], the result holds.", num_sources=3)
    assert len(problems) == 1
    assert "[5]" in problems[0]


def test_validate_citations_passes_when_in_range() -> None:
    answer = "Both [1] and [3] support this, and [2] adds nuance."
    assert validate_citations(answer, num_sources=3) == []


def test_validate_citations_no_markers_is_valid() -> None:
    assert validate_citations("No citations here.", num_sources=2) == []


def test_validate_citations_zero_sources_flags_any_marker() -> None:
    problems = validate_citations("See [1].", num_sources=0)
    assert len(problems) == 1


def test_validate_citations_reports_each_bad_index_once() -> None:
    problems = validate_citations("[5] again [5] and [9]", num_sources=3)
    assert len(problems) == 2
