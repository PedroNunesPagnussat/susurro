"""Unit tests for the daemon state machine (`susurro.daemon.Daemon`).

The socket shell (`serve`) needs a live socket, but the idle<->recording state
machine is pure and dependency-injected, so it exercises fully with fakes: a fake
recorder, a fake engine, an inject spy, and a fake clock. No mic, model, socket,
or wall-clock time.
"""

import numpy as np
import pytest

from susurro.daemon import Daemon


class FakeRecorder:
    """Stand-in for `audio.Recorder`: tracks start/stop and hands back canned audio."""

    def __init__(self, audio=None):
        self._audio = np.array([0.1, 0.2], dtype=np.float32) if audio is None else audio
        self.recording = False
        self.starts = 0
        self.stops = 0

    def start(self):
        assert not self.recording, "recorder double start"
        self.recording = True
        self.starts += 1

    def stop(self):
        assert self.recording, "recorder stop without start"
        self.recording = False
        self.stops += 1
        return self._audio


class FakeEngine:
    """Stand-in for `engine.Engine`: returns canned text, records the audio it saw."""

    def __init__(self, text="hello world"):
        self.text = text
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _make(recorder=None, engine=None, max_record_s=30.0, clock=None):
    recorder = recorder or FakeRecorder()
    engine = engine or FakeEngine()
    injected: list[str] = []
    daemon = Daemon(
        recorder,
        engine,
        injected.append,
        max_record_s=max_record_s,
        clock=clock or FakeClock(),
        log=lambda _msg: None,  # silence logging in tests
    )
    return daemon, recorder, engine, injected


# --- happy path ------------------------------------------------------------

def test_start_then_stop_transcribes_and_injects():
    daemon, recorder, engine, injected = _make()
    daemon.start()
    assert daemon.recording is True

    text = daemon.stop()
    assert text == "hello world"
    assert daemon.recording is False
    assert recorder.starts == 1 and recorder.stops == 1
    assert len(engine.calls) == 1  # transcribed the captured audio
    np.testing.assert_array_equal(engine.calls[0], recorder._audio)
    assert injected == ["hello world"]


def test_empty_transcript_is_not_injected():
    daemon, recorder, engine, injected = _make(engine=FakeEngine(text=""))
    daemon.start()
    text = daemon.stop()
    assert text == ""
    assert len(engine.calls) == 1  # still transcribed
    assert injected == []  # but nothing typed


# --- no-op / robustness ----------------------------------------------------

def test_stop_without_start_is_noop():
    daemon, recorder, engine, injected = _make()
    text = daemon.stop()
    assert text == ""
    assert recorder.stops == 0
    assert engine.calls == []
    assert injected == []
    assert daemon.recording is False


def test_duplicate_start_restarts_capture():
    daemon, recorder, _engine, _injected = _make()
    daemon.start()
    daemon.start()  # missed release then re-press: discard orphan, start fresh
    assert daemon.recording is True
    assert recorder.starts == 2  # opened twice
    assert recorder.stops == 1  # orphan torn down once
    assert recorder.recording is True  # exactly one live capture


def test_stop_resets_to_idle_even_if_engine_raises():
    class BoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("boom")

    daemon, recorder, _engine, injected = _make(engine=BoomEngine())
    daemon.start()
    with pytest.raises(RuntimeError, match="boom"):
        daemon.stop()
    assert daemon.recording is False  # not wedged
    assert recorder.stops == 1  # mic was released
    assert injected == []


# --- safety timeout --------------------------------------------------------

def test_check_timeout_does_not_fire_before_deadline():
    clock = FakeClock()
    daemon, _recorder, _engine, injected = _make(max_record_s=30.0, clock=clock)
    daemon.start()
    clock.advance(29.9)
    assert daemon.check_timeout() is None
    assert daemon.recording is True
    assert injected == []


def test_check_timeout_auto_stops_at_deadline():
    clock = FakeClock()
    daemon, recorder, _engine, injected = _make(max_record_s=30.0, clock=clock)
    daemon.start()
    clock.advance(30.0)
    text = daemon.check_timeout()
    assert text == "hello world"  # auto-stop transcribes + injects like a real stop
    assert daemon.recording is False
    assert recorder.stops == 1
    assert injected == ["hello world"]


def test_check_timeout_noop_when_idle():
    daemon, _recorder, _engine, _injected = _make()
    assert daemon.check_timeout() is None


def test_late_stop_after_auto_stop_is_noop():
    clock = FakeClock()
    daemon, recorder, engine, injected = _make(max_record_s=30.0, clock=clock)
    daemon.start()
    clock.advance(30.0)
    daemon.check_timeout()  # safety auto-stop fires
    # The physical release finally arrives after the auto-stop: must be a no-op.
    text = daemon.stop()
    assert text == ""
    assert recorder.stops == 1  # not stopped again
    assert len(engine.calls) == 1  # not transcribed again
    assert injected == ["hello world"]  # only the auto-stop injected


# --- remaining -------------------------------------------------------------

def test_remaining_counts_down_and_is_none_when_idle():
    clock = FakeClock()
    daemon, _recorder, _engine, _injected = _make(max_record_s=30.0, clock=clock)
    assert daemon.remaining() is None  # idle
    daemon.start()
    assert daemon.remaining() == pytest.approx(30.0)
    clock.advance(10.0)
    assert daemon.remaining() == pytest.approx(20.0)
    clock.advance(100.0)  # past the deadline -> clamped, never negative
    assert daemon.remaining() == 0.0
