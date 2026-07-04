"""Unit tests for the rule-based Phase-1 formatter.

Per the plan's Testing Decisions this is the highest-value automated test: pure,
deterministic, no I/O. Behaviours under test — whitespace trim/collapse, empty /
no-speech results dropped, and *no* filler-word removal (explicitly out of scope).
"""

import pytest

from susurro.formatter import RuleBasedFormatter


@pytest.fixture
def fmt() -> RuleBasedFormatter:
    return RuleBasedFormatter()


def test_passes_clean_text_through(fmt):
    assert fmt.format("hello world") == "hello world"


def test_strips_leading_and_trailing_whitespace(fmt):
    assert fmt.format("  hello world  ") == "hello world"


def test_collapses_internal_runs_of_whitespace(fmt):
    assert fmt.format("hello    world") == "hello world"


def test_collapses_newlines_and_tabs_to_single_space(fmt):
    assert fmt.format("hello\n\tworld\n\nagain") == "hello world again"


def test_empty_string_is_dropped(fmt):
    assert fmt.format("") == ""


def test_whitespace_only_is_dropped(fmt):
    assert fmt.format("   \n\t  ") == ""


def test_does_not_remove_filler_or_short_words(fmt):
    # "the sum" != filler; regex filler-removal was explicitly ruled out for Phase 1.
    text = "um the sum of you know the parts"
    assert fmt.format(text) == "um the sum of you know the parts"


def test_preserves_internal_punctuation(fmt):
    text = "And so, my fellow Americans, ask not."
    assert fmt.format(text) == "And so, my fellow Americans, ask not."


def test_none_is_treated_as_empty(fmt):
    # Whisper can hand back None for a no-speech segment.
    assert fmt.format(None) == ""
