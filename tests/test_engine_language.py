"""Unit tests for the language-code check (`susurro.engine.is_supported_language`).

The daemon uses this to refuse a bad `lang xx` at the moment it's typed, while the
model may not even be loaded — so the check reads faster-whisper's *static* list
(`faster_whisper.tokenizer._LANGUAGE_CODES`), a private name. Two things matter
here: it agrees with what Whisper actually accepts, and it fails **open** if that
private name ever moves — degrading to today's late model-side error beats breaking
dictation outright. No model or CUDA is touched.
"""

import sys
from types import SimpleNamespace

import pytest

from susurro import engine as engine_mod
from susurro.engine import is_supported_language


@pytest.fixture(autouse=True)
def _clear_language_cache():
    """The code list is memoized (`lru_cache`), so drop it around every test — else
    a real lookup would leak into the degrade test and vice versa."""
    engine_mod._language_codes.cache_clear()
    yield
    engine_mod._language_codes.cache_clear()


@pytest.mark.parametrize("code", ["en", "pt", "es", "fr", "de", "ja"])
def test_real_whisper_codes_are_accepted(code):
    assert is_supported_language(code) is True


@pytest.mark.parametrize("code", ["xx", "zz", "english", "", "en-US"])
def test_bogus_codes_are_rejected(code):
    # These are exactly what Whisper's tokenizer raises ValueError on, one utterance
    # too late; the point of the check is to catch them at the switch instead.
    assert is_supported_language(code) is False


def test_matching_is_exact_like_whispers_own_check():
    # faster-whisper does not lower-case the code before its membership test, so
    # neither do we — accepting "EN" here would just move the failure back into the
    # model.
    assert is_supported_language("EN") is False


def test_missing_private_name_degrades_to_accepting_anything(monkeypatch):
    # A future faster-whisper that renames/moves `_LANGUAGE_CODES` must not break
    # dictation: validation goes permissive and Whisper's own error is the backstop.
    monkeypatch.setitem(sys.modules, "faster_whisper.tokenizer", SimpleNamespace())
    assert engine_mod._language_codes() is None
    assert is_supported_language("xx") is True
    assert is_supported_language("en") is True


def test_unusable_code_list_degrades_to_accepting_anything(monkeypatch):
    # Same guard one level down: the name still exists but is no longer iterable.
    monkeypatch.setitem(
        sys.modules, "faster_whisper.tokenizer", SimpleNamespace(_LANGUAGE_CODES=None)
    )
    assert engine_mod._language_codes() is None
    assert is_supported_language("xx") is True


def test_code_list_is_read_only_once(monkeypatch):
    # The check runs on a key-press path, so the (import + set build) is memoized.
    assert is_supported_language("en") is True
    monkeypatch.setitem(sys.modules, "faster_whisper.tokenizer", SimpleNamespace())
    assert is_supported_language("xx") is False  # cached list, not re-imported
