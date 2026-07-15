"""Unit tests for the hardware-free parts of `susurro-bench record`.

The mic capture and the interactive prompts need a person and a device, but the
two pieces of real logic are pure: the WAV writer (round-trips through `load_wav`,
prior art `tests/test_audio.py`) and the planner that decides which scripts to
record given what's already on disk and the flags.
"""

import numpy as np
import pytest

from susurro.audio import SAMPLE_RATE, load_wav
from susurro.bench.recording import (
    plan_recordings,
    recorded_ids,
    save_wav,
    script_ids,
)

# --- save_wav (round-trip through load_wav) --------------------------------


def test_save_wav_roundtrips_through_load_wav(tmp_path):
    # Values that are exactly representable at 16-bit (multiples of 1/32768) so the
    # round-trip is lossless where it can be; the writer must be load_wav's inverse.
    audio = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    path = tmp_path / "clip.wav"
    save_wav(path, audio)

    back = load_wav(path)
    assert back.dtype == np.float32
    assert back.ndim == 1
    np.testing.assert_allclose(back, audio, atol=1.0 / 32768)


def test_save_wav_writes_mono_at_the_given_sample_rate(tmp_path):
    import wave

    path = tmp_path / "clip.wav"
    save_wav(path, np.zeros(800, dtype=np.float32), sample_rate=SAMPLE_RATE)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16-bit PCM, what load_wav expects
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == 800


def test_save_wav_clips_out_of_range_samples(tmp_path):
    # A hot mic can clip past [-1, 1]; the writer must clamp, not wrap around to the
    # opposite sign via int16 overflow.
    path = tmp_path / "clip.wav"
    save_wav(path, np.array([1.5, -1.5], dtype=np.float32))
    back = load_wav(path)
    assert back[0] == pytest.approx(1.0, abs=1.0 / 32768)
    assert back[1] == pytest.approx(-1.0, abs=1.0 / 32768)


# --- planner (which ids to record) -----------------------------------------


def test_plan_default_skips_already_recorded_ids():
    assert plan_recordings(["a", "b", "c"], {"b"}) == ["a", "c"]


def test_plan_default_records_nothing_when_all_present():
    assert plan_recordings(["a", "b"], {"a", "b"}) == []


def test_plan_redo_rerecords_everything_in_order():
    assert plan_recordings(["a", "b", "c"], {"a", "b"}, redo=True) == ["a", "b", "c"]


def test_plan_only_records_just_that_id_even_if_already_recorded():
    assert plan_recordings(["a", "b", "c"], {"b"}, only="b") == ["b"]


def test_plan_only_unknown_id_raises_with_the_id():
    with pytest.raises(KeyError) as exc:
        plan_recordings(["a", "b"], set(), only="zzz")
    assert "zzz" in str(exc.value)


# --- discovery helpers -----------------------------------------------------


def test_script_ids_are_txt_stems_sorted(tmp_path):
    (tmp_path / "02-two.txt").write_text("two")
    (tmp_path / "01-one.txt").write_text("one")
    (tmp_path / "notes.md").write_text("ignore me")  # non-txt ignored
    assert script_ids(tmp_path) == ["01-one", "02-two"]


def test_recorded_ids_are_wav_stems(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "c.txt").write_text("nope")
    assert recorded_ids(tmp_path) == {"a", "b"}


def test_recorded_ids_missing_dir_is_empty(tmp_path):
    assert recorded_ids(tmp_path / "does-not-exist") == set()
