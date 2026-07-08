"""Unit tests for the daemon state machine (`susurro.daemon.Daemon`).

The socket shell (`serve`) needs a live socket, but the idle<->recording state
machine is pure and dependency-injected, so it exercises fully with fakes: a fake
recorder, a fake engine, an inject spy, and a fake clock. No mic, model, socket,
or wall-clock time.
"""

import argparse

import numpy as np
import pytest

from susurro.config import Config, DaemonConfig, EngineConfig
from susurro.daemon import Daemon, _apply_cli, _dispatch


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
        self.language = "en"

    def set_language(self, code):
        self.language = code

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
        self.language = "en"
        self._loaded = True  # warmed at startup, like the real daemon
        self.unloads = 0
        self.loads = 0

    @property
    def loaded(self):
        return self._loaded

    def set_language(self, code):
        self.language = code

    def load(self):
        self._loaded = True
        self.loads += 1

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
    """Stand-in for `notify.Notifier`: records the (state, arg) sequence so tests
    can assert the recording toast is raised on start (with the active language)
    and always cleared on stop, and the language toast fires on a switch."""

    def __init__(self):
        self.events = []  # ("recording", lang) / ("done", text) / ("language", code)

    def recording(self, language):
        self.events.append(("recording", language))

    def done(self, text):
        self.events.append(("done", text))

    def language(self, code):
        self.events.append(("language", code))


def _make(
    recorder=None, engine=None, max_record_s=30.0, clock=None, idle_timeout_s=None, language="en"
):
    recorder = recorder or FakeRecorder()
    engine = engine or FakeEngine()
    injected: list[str] = []
    daemon = Daemon(
        recorder,
        engine,
        injected.append,
        notify=SpyNotifier(),
        language=language,
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


# --- preload on key-press --------------------------------------------------

def test_press_reloads_an_unloaded_managed_engine():
    # After an idle-unload, the press must rebuild the model *now* so the load
    # overlaps the hold — engine.loaded is True before any stop() runs.
    daemon, engine, clock, _injected = _make_managed(idle_timeout_s=60.0)
    clock.advance(60.0)
    daemon.check_idle()  # idle-unload drops the model
    assert engine.loaded is False

    daemon.start()
    assert engine.loaded is True  # rebuilt on the press, not deferred to stop
    assert engine.loads == 1
    assert daemon.recording is True


def test_press_does_not_reload_an_already_loaded_engine():
    # A warm engine must not be rebuilt on the press — no double-load.
    daemon, engine, _clock, _injected = _make_managed(idle_timeout_s=60.0)
    assert engine.loaded is True

    daemon.start()
    assert engine.loads == 0  # already warm — load() never called
    assert engine.loaded is True


def test_press_on_plain_engine_never_attempts_load():
    # A non-managed engine has no load(); the press must not touch it (a wrongful
    # call would AttributeError on the plain FakeEngine). Behaves exactly as today.
    daemon, recorder, _engine, injected = _make(engine=FakeEngine())
    daemon.start()  # must not raise
    assert daemon.recording is True
    assert recorder.starts == 1

    daemon.stop()  # and the full round-trip still works
    assert injected == ["hello world"]


def test_failed_preload_leaves_daemon_recording_and_stop_still_transcribes():
    # Preload is best-effort: a load() that raises must not abort the capture,
    # and the release-path transcribe still reloads + injects.
    class FlakyPreloadEngine(FakeManagedEngine):
        def load(self):
            self.loads += 1
            raise RuntimeError("CUDA OOM on preload")

    engine = FlakyPreloadEngine()
    engine._loaded = False  # start unloaded, as after an idle-unload
    daemon, recorder, _engine, injected = _make(engine=engine)

    daemon.start()  # preload raises internally
    assert daemon.recording is True  # capture survived the failed preload
    assert recorder.recording is True
    assert engine.loads == 1  # preload was attempted

    text = daemon.stop()  # transcribe-path reload picks up the slack
    assert text == "hello world"
    assert daemon.recording is False
    assert injected == ["hello world"]


def test_short_press_then_immediate_stop_still_transcribes():
    # A press immediately followed by a release (no unload in between) must
    # transcribe, inject, and return to idle without wedging.
    daemon, engine, _clock, injected = _make_managed(idle_timeout_s=60.0)
    daemon.start()
    text = daemon.stop()
    assert text == "hello world"
    assert daemon.recording is False
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
    assert daemon._notify.events == [("recording", "en")]


def test_stop_clears_toast_with_transcript():
    daemon, _recorder, _engine, _injected = _make()
    daemon.start()
    daemon.stop()
    assert daemon._notify.events == [("recording", "en"), ("done", "hello world")]


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
    assert daemon._notify.events == [("recording", "en"), ("done", "hello world")]


def test_toast_is_cleared_even_if_engine_raises():
    class BoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("boom")

    daemon, _recorder, _engine, _injected = _make(engine=BoomEngine())
    daemon.start()
    with pytest.raises(RuntimeError, match="boom"):
        daemon.stop()
    # the persistent -t 0 toast must still be replaced on failure, else it hangs
    assert daemon._notify.events == [("recording", "en"), ("done", "")]


# --- language --------------------------------------------------------------

def test_set_language_updates_engine_and_posts_toast():
    daemon, _recorder, engine, _injected = _make()
    active = daemon.set_language("pt")
    assert active == "pt"
    assert daemon.language == "pt"
    assert engine.language == "pt"  # pushed to the engine, no reload
    assert daemon._notify.events == [("language", "pt")]


def test_toggle_flips_en_and_pt():
    daemon, _recorder, engine, _injected = _make(language="en")
    assert daemon.set_language("toggle") == "pt"
    assert engine.language == "pt"
    assert daemon.set_language("toggle") == "en"  # flips back
    assert engine.language == "en"
    assert daemon._notify.events == [("language", "pt"), ("language", "en")]


def test_start_uses_the_active_language_in_recording_toast():
    daemon, _recorder, _engine, _injected = _make(language="en")
    daemon.set_language("pt")
    daemon.start()
    assert daemon._notify.events == [("language", "pt"), ("recording", "pt")]


def test_daemon_boots_into_the_given_language():
    daemon, _recorder, _engine, _injected = _make(language="pt")
    daemon.start()
    assert daemon._notify.events == [("recording", "pt")]


# --- dispatch --------------------------------------------------------------

def test_dispatch_lang_with_code_sets_explicit():
    daemon, _recorder, engine, _injected = _make(language="en")
    _dispatch(daemon, "lang pt")
    assert daemon.language == "pt"
    assert engine.language == "pt"


def test_dispatch_bare_lang_toggles():
    daemon, _recorder, _engine, _injected = _make(language="en")
    _dispatch(daemon, "lang")
    assert daemon.language == "pt"


def test_dispatch_start_and_stop_still_route():
    daemon, recorder, _engine, injected = _make()
    _dispatch(daemon, "start")
    assert daemon.recording is True
    _dispatch(daemon, "stop")
    assert daemon.recording is False
    assert recorder.stops == 1


# --- CLI over config precedence (_apply_cli) -------------------------------

def _args(**over):
    """A parsed-args stand-in: every flag defaults to its "not passed" sentinel."""
    base = dict(
        model=None, device=None, cpu=False, lang=None,
        max_record=None, idle_timeout=None, no_notify=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_apply_cli_no_flags_leaves_config_untouched():
    config = Config()
    assert _apply_cli(config, _args()) == config


def test_apply_cli_flags_override_config_values():
    config = Config()
    out = _apply_cli(config, _args(model="medium", lang="pt", device=3, max_record=90.0, idle_timeout=0.0))
    assert out.engine.model == "medium"
    assert out.engine.language == "pt"
    assert out.audio.device == 3
    assert out.daemon.max_record_s == 90.0
    assert out.daemon.idle_timeout_s == 0.0
    # untouched fields keep the config value
    assert out.engine.beam_size == 5
    assert out.engine.compute_type == "int8"


def test_apply_cli_config_values_survive_when_no_flag():
    # A non-default config with every flag unset -> config wins over built-ins.
    config = Config(
        engine=EngineConfig(model="tiny", language="pt", beam_size=2),
        daemon=DaemonConfig(max_record_s=120.0),
    )
    out = _apply_cli(config, _args())
    assert out.engine.model == "tiny"
    assert out.engine.language == "pt"
    assert out.engine.beam_size == 2
    assert out.daemon.max_record_s == 120.0


def test_cpu_flag_forces_cpu_over_config_cuda():
    config = Config(engine=EngineConfig(device="cuda"))
    assert _apply_cli(config, _args(cpu=True)).engine.device == "cpu"


def test_no_notify_flag_forces_notifications_off():
    config = Config(daemon=DaemonConfig(notify=True))
    assert _apply_cli(config, _args(no_notify=True)).daemon.notify is False


def test_config_can_disable_notifications_without_the_flag():
    config = Config(daemon=DaemonConfig(notify=False))
    assert _apply_cli(config, _args(no_notify=False)).daemon.notify is False
