"""Unit tests for the hardware-free parts of `susurro-bench record`.

The mic capture and the interactive prompts need a person and a device, but the
real logic is pure: the WAV writer (round-trips through `load_wav`, prior art
`tests/test_audio.py`), the planner that decides which scripts to record given
what's already on disk and the flags, and the `[audio]` resolution that decides
*which input* gets captured (defaults < config file < CLI flag).
"""

import argparse

import numpy as np
import pytest

from susurro.audio import SAMPLE_RATE, load_wav
from susurro.bench.recording import (
    plan_recordings,
    record_command,
    recorded_ids,
    resolve_audio,
    save_wav,
    script_ids,
)
from susurro.config import ConfigError

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


# --- [audio] resolution (defaults < config file < CLI flag) ----------------


def _args(tmp_path, *, toml: str = "", device: str | None = None) -> argparse.Namespace:
    """A `record` namespace pointed at a throwaway config file, so these tests never
    read (or depend on) the repo's own config.toml."""
    path = tmp_path / "config.toml"
    path.write_text(toml)
    return argparse.Namespace(config=str(path), device=device)


def test_resolve_audio_uses_the_config_device_when_no_flag(tmp_path):
    # The bug this closes: `record` used to ignore [audio] entirely and capture ten
    # silent takes from the default input the owner had already configured around.
    assert resolve_audio(_args(tmp_path, toml="[audio]\ndevice = 4\n")).device == 4


def test_resolve_audio_device_flag_overrides_the_config(tmp_path):
    aud = resolve_audio(_args(tmp_path, toml="[audio]\ndevice = 4\n", device="7"))
    assert aud.device == 7  # digits -> PortAudio index, same rule as the other CLIs


def test_resolve_audio_device_flag_accepts_a_name_substring(tmp_path):
    aud = resolve_audio(_args(tmp_path, toml="[audio]\ndevice = 4\n", device="Yeti"))
    assert aud.device == "Yeti"


def test_resolve_audio_falls_back_to_built_in_defaults(tmp_path):
    aud = resolve_audio(_args(tmp_path))  # config file present but empty
    assert aud.device is None  # PortAudio's default input
    assert aud.sample_rate == SAMPLE_RATE


def test_resolve_audio_rejects_a_sample_rate_the_harness_cannot_benchmark(tmp_path):
    # No backend resamples, so a non-16k rate is refused before `record` writes ten
    # takes that `run` would only reject later. The check now lives in `config` (one
    # gate for every entrypoint, daemon included); this pins that `record` inherits it.
    with pytest.raises(ConfigError) as exc:
        resolve_audio(_args(tmp_path, toml="[audio]\nsample_rate = 48000\n"))
    assert "48000" in str(exc.value) and str(SAMPLE_RATE) in str(exc.value)


def test_record_command_reports_a_config_error_before_prompting(capsys, tmp_path):
    # The namespace deliberately carries only the flags resolve_audio needs: if the
    # config check ever moves after the planner, this raises instead of blocking the
    # suite on the interactive `input()`.
    rc = record_command(_args(tmp_path, toml="[audio]\nsample_rate = 48000\n"))
    assert rc == 1
    assert "48000" in capsys.readouterr().err
