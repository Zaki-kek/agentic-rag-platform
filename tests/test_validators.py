"""Tests for the extra neutral quality gates and their helpers."""

from __future__ import annotations

from app.generation.validators import (
    MinLengthGate,
    PlaceholderGate,
    ReferencesGate,
    find_placeholders,
    references_present,
)


def test_find_placeholders_ignores_normal_text() -> None:
    assert find_placeholders("A perfectly normal sentence about a todo list.") == []
    assert find_placeholders("") == []
    assert find_placeholders("Use the xxx-large size of curly braces { and }.") == []


def test_find_placeholders_flags_mustache_and_brackets() -> None:
    found = find_placeholders("Hello {{name}}, please fix [TODO] and [TBD] soon.")
    assert "{{name}}" in found
    assert "[TODO]" in found
    assert "[TBD]" in found


def test_find_placeholders_flags_bare_markers_case_sensitively() -> None:
    found = find_placeholders("Section status: TBD. Also FIXME and XXX remain.")
    assert "TBD" in found
    assert "FIXME" in found
    assert "XXX" in found
    # Lowercase variants are ordinary prose and must not be flagged.
    assert find_placeholders("the fixme report") == []


def test_find_placeholders_preserves_order_and_duplicates() -> None:
    found = find_placeholders("{{a}} then [TODO] then {{b}} then [TODO]")
    assert found == ["{{a}}", "[TODO]", "{{b}}", "[TODO]"]


def test_placeholder_gate_pass() -> None:
    gate = PlaceholderGate("draft")
    result = gate.check({"draft": "A finished paragraph with no markers."})
    assert result.passed
    assert gate.name == "no_placeholders:draft"


def test_placeholder_gate_fail_lists_placeholders() -> None:
    gate = PlaceholderGate("draft")
    result = gate.check({"draft": "Intro {{topic}} with [TODO]."})
    assert not result.passed
    assert "{{topic}}" in result.reason
    assert "[TODO]" in result.reason


def test_placeholder_gate_missing_field_passes() -> None:
    assert PlaceholderGate("draft").check({}).passed


def test_min_length_gate_pass() -> None:
    gate = MinLengthGate("draft", 5)
    result = gate.check({"draft": "long enough"})
    assert result.passed
    assert gate.name == "min_length:draft>=5"


def test_min_length_gate_fail() -> None:
    gate = MinLengthGate("draft", 10)
    result = gate.check({"draft": "short"})
    assert not result.passed
    assert "too short" in result.reason


def test_min_length_gate_missing_field_fails_when_required() -> None:
    assert not MinLengthGate("draft", 1).check({}).passed


def test_references_present_helper() -> None:
    assert references_present("See [1] and [2] for details.", min_refs=2)
    assert not references_present("See [1] only.", min_refs=2)
    assert not references_present("No citations at all.")
    assert references_present("nothing required", min_refs=0)


def test_references_gate_pass() -> None:
    gate = ReferencesGate("draft", min_refs=1)
    result = gate.check({"draft": "As shown in [1], the method works."})
    assert result.passed
    assert gate.name == "references:draft>=1"


def test_references_gate_fail() -> None:
    gate = ReferencesGate("draft", min_refs=2)
    result = gate.check({"draft": "Only one source here [1]."})
    assert not result.passed
    assert "citation marker" in result.reason
