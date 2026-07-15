"""Unit tests for the WER/CER scorer — the pure heart of the benchmark's accuracy
numbers. Expected values are hand-computed from the (reference, hypothesis) pair,
not read back from the code. Prior art: `tests/test_formatter.py`.

Skipped when the `bench` extra (jiwer) isn't installed, so a core-only checkout
still has a green `pytest`.
"""

import pytest

pytest.importorskip("jiwer")

from susurro.bench.wer import Score, score  # noqa: E402  (after importorskip)


def test_identical_text_scores_zero_on_every_metric():
    s = score("The cat sat on the mat.", "The cat sat on the mat.")
    assert s == Score(norm_wer=0.0, raw_wer=0.0, cer=0.0)


def test_one_substitution_in_ten_words_is_point_one_wer():
    # ten reference words, "ten" -> "zero": a single real substitution. No case or
    # punctuation difference, so raw and normalized agree at 0.1.
    ref = "one two three four five six seven eight nine ten"
    hyp = "one two three four five six seven eight nine zero"
    s = score(ref, hyp)
    assert s.norm_wer == pytest.approx(0.1)
    assert s.raw_wer == pytest.approx(0.1)


def test_case_and_punctuation_only_diff_is_zero_normalized_but_hurts_raw():
    # Both words differ only by case/punctuation: normalization erases it (0.0),
    # but raw WER counts both as substitutions (2/2 = 1.0) — "what gets typed".
    s = score("Hello, world!", "hello world")
    assert s.norm_wer == pytest.approx(0.0)
    assert s.raw_wer == pytest.approx(1.0)


def test_normalization_lowercases_before_comparing():
    # One of two words differs only by case -> normalized 0.0, raw 0.5.
    s = score("ABC def", "abc def")
    assert s.norm_wer == pytest.approx(0.0)
    assert s.raw_wer == pytest.approx(0.5)


def test_cer_is_finer_grained_than_wer():
    # A one-character slip inside a single word is a whole wrong word (WER 1.0) but
    # only one of five characters (CER 0.2) — CER de-noises the word boundary.
    s = score("abcde", "abfde")
    assert s.norm_wer == pytest.approx(1.0)
    assert s.cer == pytest.approx(0.2)


def test_empty_hypothesis_is_total_error():
    # The model returned nothing: every reference word/char is a deletion -> 1.0.
    s = score("one two", "")
    assert s.norm_wer == pytest.approx(1.0)
    assert s.raw_wer == pytest.approx(1.0)
    assert s.cer == pytest.approx(1.0)
