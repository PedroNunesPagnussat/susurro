"""Unit tests for the hardware-free parts of the capture module.

The PortAudio stream needs a mic, but the block-assembly (`_WindowBuffer`) and WAV
loading are pure and get exercised here without any device.
"""

from pathlib import Path

import numpy as np
import pytest

from susurro.audio import SAMPLE_RATE, Recorder, _WindowBuffer, load_wav

FIXTURE = Path(__file__).parent / "fixtures" / "jfk_16k_mono.wav"


# --- _WindowBuffer ---------------------------------------------------------


def test_buffer_concatenates_mono_blocks_in_order():
    buf = _WindowBuffer()
    buf.add(np.array([[0.1], [0.2]], dtype=np.float32))
    buf.add(np.array([[0.3]], dtype=np.float32))
    np.testing.assert_allclose(buf.result(), [0.1, 0.2, 0.3], rtol=1e-6)


def test_buffer_takes_first_channel_of_multichannel_block():
    stereo = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    buf = _WindowBuffer()
    buf.add(stereo)
    np.testing.assert_allclose(buf.result(), [0.1, 0.2], rtol=1e-6)


def test_buffer_trims_to_construction_cap():
    buf = _WindowBuffer(max_samples=4)
    buf.add(np.arange(10, dtype=np.float32).reshape(-1, 1))
    assert buf.result().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_buffer_result_is_float32():
    buf = _WindowBuffer()
    buf.add(np.array([[1], [2]], dtype=np.float32))
    assert buf.result().dtype == np.float32


def test_empty_buffer_returns_empty_float32_array():
    out = _WindowBuffer().result()
    assert out.dtype == np.float32 and out.shape == (0,)


def test_buffer_counts_xrun_status():
    buf = _WindowBuffer()
    buf.add(np.zeros((2, 1), dtype=np.float32), status="input overflow")
    buf.add(np.zeros((2, 1), dtype=np.float32), status=None)
    assert buf.xruns == 1


def test_buffer_cap_drops_blocks_once_full():
    buf = _WindowBuffer(max_samples=3)
    buf.add(np.array([[0.1], [0.2]], dtype=np.float32))  # 2 samples, under cap
    assert not buf.capped
    buf.add(np.array([[0.3], [0.4]], dtype=np.float32))  # crosses cap, still stashed
    assert not buf.capped
    buf.add(np.array([[0.5]], dtype=np.float32))  # already full -> dropped
    assert buf.capped
    # result() defaults to the construction-time cap -> trimmed to 3 samples
    np.testing.assert_allclose(buf.result(), [0.1, 0.2, 0.3], rtol=1e-6)


def test_buffer_cap_none_keeps_everything():
    buf = _WindowBuffer(max_samples=None)
    buf.add(np.arange(5, dtype=np.float32).reshape(-1, 1))
    assert not buf.capped
    assert buf.result().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- Recorder (hardware-free guards; start/stop need a mic) ----------------


def test_recorder_not_recording_initially():
    assert Recorder().recording is False


def test_recorder_stop_without_start_raises():
    with pytest.raises(RuntimeError, match="not recording"):
        Recorder().stop()


def _fake_sounddevice(monkeypatch):
    """Inject a fake `sounddevice` so Recorder.start() opens no real PortAudio
    stream; returns the stubbed `InputStream` to inspect the kwargs it was given."""
    import sys
    import types
    from unittest import mock

    fake = types.SimpleNamespace()
    fake.PortAudioError = type("PortAudioError", (Exception,), {})
    fake.InputStream = mock.Mock(return_value=mock.Mock())
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake.InputStream


def test_recorder_opens_stream_with_configured_channels_and_rate(monkeypatch):
    InputStream = _fake_sounddevice(monkeypatch)
    Recorder(samplerate=48_000, channels=2).start()
    kwargs = InputStream.call_args.kwargs
    assert kwargs["channels"] == 2
    assert kwargs["samplerate"] == 48_000


def test_recorder_defaults_to_mono_16k(monkeypatch):
    InputStream = _fake_sounddevice(monkeypatch)
    Recorder().start()
    kwargs = InputStream.call_args.kwargs
    assert kwargs["channels"] == 1
    assert kwargs["samplerate"] == SAMPLE_RATE


def test_recorder_roundtrip_assembles_callback_audio(monkeypatch):
    # start() -> PortAudio drives the callback -> stop() returns the assembled mono
    # buffer. Exercises the Recorder wiring end to end (not just _WindowBuffer): the
    # callback closure feeds `buf`, and stop() reads it back.
    InputStream = _fake_sounddevice(monkeypatch)
    rec = Recorder()
    rec.start()
    callback = InputStream.call_args.kwargs["callback"]
    callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
    callback(np.array([[0.3]], dtype=np.float32), 1, None, None)
    out = rec.stop()
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3], rtol=1e-6)
    assert rec.recording is False  # stream released


def test_recorder_closes_stream_if_start_raises(monkeypatch):
    # If open() succeeds but start() fails, the allocated stream must be closed
    # (not leaked) and the error surfaced as RuntimeError.
    fake = _fake_sounddevice(monkeypatch)
    import sounddevice as sd  # the fake just injected into sys.modules

    stream = fake.return_value
    stream.start.side_effect = sd.PortAudioError("stream start failed")
    with pytest.raises(RuntimeError, match="audio capture failed"):
        Recorder().start()
    stream.close.assert_called_once()  # no leak on the error path


# --- load_wav --------------------------------------------------------------


def test_load_wav_returns_mono_float32_in_unit_range():
    audio = load_wav(FIXTURE)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.abs(audio).max() <= 1.0
    # ~11s at 16kHz
    assert abs(len(audio) - 11 * SAMPLE_RATE) < SAMPLE_RATE


def test_load_wav_downmixes_multichannel_to_first_channel(tmp_path):
    # A stereo WAV must come back mono (first channel only), not interleaved: 3
    # frames -> 3 samples, and the left channel (not the right) is what survives.
    import wave

    path = tmp_path / "stereo.wav"
    left, right = 16_000, -16_000  # distinct per channel so a wrong pick is visible
    frames = np.array([[left, right]] * 3, dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(frames)

    out = load_wav(path)
    assert out.ndim == 1
    assert out.shape == (3,)  # 3 frames, mono — not 6 interleaved samples
    np.testing.assert_allclose(out, [left / 32768.0] * 3, rtol=1e-6)  # left kept


def _write_wav(path, rate: int) -> None:
    """A silent 16-bit mono WAV at `rate` — just a header to load against."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(10, dtype=np.int16).tobytes())


def test_load_wav_rejects_a_wav_at_another_sample_rate(tmp_path):
    # Nothing here resamples, so a 48kHz take read as 16kHz would transcribe as
    # garbage *and* be timed against 3x its real duration — both silently. The
    # error has to name the file and both rates to be actionable.
    path = tmp_path / "48k.wav"
    _write_wav(path, 48_000)

    with pytest.raises(ValueError) as exc:
        load_wav(path)

    message = str(exc.value)
    assert "48000" in message and str(SAMPLE_RATE) in message
    assert "48k.wav" in message


def test_load_wav_names_the_file_on_a_truncated_data_chunk(tmp_path):
    # Ctrl-C during `save_wav` leaves an odd number of bytes in the data chunk;
    # numpy then raises "buffer size must be a multiple of element size" with no
    # filename, and the bench turned that into a warning naming no recording.
    path = tmp_path / "truncated.wav"
    _write_wav(path, SAMPLE_RATE)
    path.write_bytes(path.read_bytes()[:-1])  # chop one byte off the samples

    with pytest.raises(ValueError) as exc:
        load_wav(path)

    message = str(exc.value)
    assert "truncated.wav" in message  # which recording, the only actionable part
    assert "multiple of element size" in message  # keeps numpy's own diagnosis


def test_load_wav_names_the_file_on_a_channel_misaligned_chunk(tmp_path):
    # Same class through the other raise: a stereo take whose frame count doesn't
    # divide by the channel count blows up in `reshape`, also unnamed.
    import wave

    path = tmp_path / "misaligned.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.zeros(4, dtype=np.int16).tobytes())
    # Drop one int16 so the sample count is odd against 2 channels.
    path.write_bytes(path.read_bytes()[:-2])

    with pytest.raises(ValueError, match="misaligned.wav"):
        load_wav(path)
