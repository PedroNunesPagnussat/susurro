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


class FakeManagedEngine:
    """Stand-in for `engine.LazyEngine`: transcribes, and can be unloaded/reloaded.

    Tracks load state so the daemon's idle-unload path can be asserted with the
    fake clock — no real model, no CUDA.
    """

    def __init__(self, text="hello world"):
        self.text = text
        self.calls = []
        self._loaded = True  # warmed at startup, like the real daemon
        self.unloads = 0

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self._loaded = False
        self.unloads += 1

    def transcribe(self, audio):
        self.calls.append(audio)
        self._loaded = True  # a real transcribe reloads if it was dropped
        return self.text


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class SpyNotifier:
    """Stand-in for `notify.Notifier`: records the (state, text) sequence so tests
    can assert the recording toast is raised on start and always cleared on stop."""

    def __init__(self):
        self.events = []  # ("recording",) / ("done", text)

    def recording(self):
        self.events.append(("recording",))

    def done(self, text):
        self.events.append(("done", text))


def _make(recorder=None, engine=None, max_record_s=30.0, clock=None, idle_timeout_s=None):
    recorder = recorder or FakeRecorder()
    engine = engine or FakeEngine()
    injected: list[str] = []
    daemon = Daemon(
        recorder,
        engine,
        injected.append,
        notify=SpyNotifier(),
        max_record_s=max_record_s,
        idle_timeout_s=idle_timeout_s,
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


# --- idle unload -----------------------------------------------------------

def _make_managed(idle_timeout_s=60.0, clock=None):
    clock = clock or FakeClock()
    engine = FakeManagedEngine()
    daemon, recorder, _engine, injected = _make(
        engine=engine, idle_timeout_s=idle_timeout_s, clock=clock
    )
    return daemon, engine, clock, injected


def test_idle_remaining_none_when_feature_disabled():
    daemon, _engine, _clock, _injected = _make_managed(idle_timeout_s=None)
    assert daemon.idle_remaining() is None
    assert daemon.check_idle() is False


def test_idle_remaining_counts_down_when_idle_and_loaded():
    daemon, _engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    assert daemon.idle_remaining() == pytest.approx(60.0)
    clock.advance(20.0)
    assert daemon.idle_remaining() == pytest.approx(40.0)


def test_idle_remaining_none_while_recording():
    daemon, _engine, _clock, _injected = _make_managed(idle_timeout_s=60.0)
    daemon.start()
    assert daemon.idle_remaining() is None  # never unload mid-capture
    assert daemon.check_idle() is False


def test_check_idle_does_not_unload_before_deadline():
    daemon, engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(59.9)
    assert daemon.check_idle() is False
    assert engine.loaded is True
    assert engine.unloads == 0


def test_check_idle_unloads_at_deadline():
    daemon, engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(60.0)
    assert daemon.check_idle() is True
    assert engine.loaded is False
    assert engine.unloads == 1
    assert daemon.idle_remaining() is None  # already unloaded -> nothing to wake for


def test_check_idle_is_noop_when_already_unloaded():
    daemon, engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(60.0)
    daemon.check_idle()  # first unload
    clock.advance(60.0)
    assert daemon.check_idle() is False  # nothing left to drop
    assert engine.unloads == 1


def test_activity_resets_the_idle_timer():
    daemon, engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(50.0)
    daemon.start()  # use resets the timer
    daemon.stop()
    assert engine.loaded is True
    clock.advance(50.0)  # 50s since stop, < 60s
    assert daemon.check_idle() is False
    clock.advance(10.0)  # now 60s idle since stop
    assert daemon.check_idle() is True


def test_transcribe_reloads_after_idle_unload():
    daemon, engine, clock, injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(60.0)
    daemon.check_idle()
    assert engine.loaded is False

    daemon.start()
    text = daemon.stop()  # pays the reload, then transcribes
    assert text == "hello world"
    assert engine.loaded is True
    assert injected == ["hello world"]


def test_plain_engine_never_unloads():
    # A non-managed engine (no loaded/unload) disables idle-unload even if a
    # timeout is set — the feature needs a managed engine.
    clock = FakeClock()
    daemon, _recorder, _engine, _injected = _make(idle_timeout_s=60.0, clock=clock)
    clock.advance(120.0)
    assert daemon.idle_remaining() is None
    assert daemon.check_idle() is False


# --- notifications ---------------------------------------------------------

def test_start_raises_recording_toast():
    daemon, _recorder, _engine, _injected = _make()
    daemon.start()
    assert daemon._notify.events == [("recording",)]


def test_stop_clears_toast_with_transcript():
    daemon, _recorder, _engine, _injected = _make()
    daemon.start()
    daemon.stop()
    assert daemon._notify.events == [("recording",), ("done", "hello world")]


def test_stop_without_start_does_not_notify():
    daemon, _recorder, _engine, _injected = _make()
    daemon.stop()  # no-op stop must not post a stray toast
    assert daemon._notify.events == []


def test_auto_stop_clears_toast():
    clock = FakeClock()
    daemon, _recorder, _engine, _injected = _make(max_record_s=30.0, clock=clock)
    daemon.start()
    clock.advance(30.0)
    daemon.check_timeout()
    assert daemon._notify.events == [("recording",), ("done", "hello world")]


def test_toast_is_cleared_even_if_engine_raises():
    class BoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("boom")

    daemon, _recorder, _engine, _injected = _make(engine=BoomEngine())
    daemon.start()
    with pytest.raises(RuntimeError, match="boom"):
        daemon.stop()
    # the persistent -t 0 toast must still be replaced on failure, else it hangs
    assert daemon._notify.events == [("recording",), ("done", "")]
