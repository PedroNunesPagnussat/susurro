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


# --- load_wav --------------------------------------------------------------

def test_load_wav_returns_mono_float32_in_unit_range():
    audio = load_wav(FIXTURE)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.abs(audio).max() <= 1.0
    # ~11s at 16kHz
    assert abs(len(audio) - 11 * SAMPLE_RATE) < SAMPLE_RATE
