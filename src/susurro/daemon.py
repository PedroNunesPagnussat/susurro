"""The warm, always-on daemon: the half of hold-to-talk that owns the model.

Holds the warm `Engine` + a `Recorder` and drives an idle<->recording state
machine over a Unix socket (tiny `start`/`stop` line protocol). The client
(`susurro-ctl`) is a trivial socket write bound to a Hyprland key; all the cost
(model load, CUDA warm) is paid once here at autostart, so per-utterance latency
is inference-bound.

On `stop` (or the safety auto-stop timeout): transcribe -> format -> inject into
the focused window. The state machine (`Daemon`) is pure and dependency-injected
— recorder, engine, inject, notify, and clock are all injectable — so it
unit-tests with fakes: no mic, model, socket, or wall-clock. `serve()` is the thin
socket shell around it.

The safety timeout is the anti-wedge backstop: a missed key release (a dropped
`stop`) must not leave the daemon recording forever.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np

from ._ipc import socket_path
from .audio import Recorder
from .config import Config, ConfigError, load_config, pick
from .engine import Engine, LazyEngine
from .inject import inject
from .notify import Notifier, NullNotifier

DEFAULT_MAX_RECORD_S = 60.0


class _Capturer(Protocol):
    """Structural type for the recorder seam (real: `audio.Recorder`)."""

    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...


class _Transcriber(Protocol):
    """Structural type for the engine seam (real: `engine.Engine`)."""

    def transcribe(self, audio: np.ndarray) -> str: ...
    def set_language(self, code: str) -> None: ...


@runtime_checkable
class _ManagedEngine(Protocol):
    """A `_Transcriber` that can also be unloaded/reloaded (real: `engine.LazyEngine`).

    Runtime-checkable so the daemon can detect at construction whether idle-unload
    is even possible: a plain engine without this capability just disables it.
    """

    @property
    def loaded(self) -> bool: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def transcribe(self, audio: np.ndarray) -> str: ...


class _Notifier(Protocol):
    """Structural type for the notification seam (real: `notify.Notifier`)."""

    def recording(self, language: str) -> None: ...
    def done(self, text: str) -> None: ...
    def language(self, code: str) -> None: ...


# Two-way language toggle (Super+Shift+D). Kept a small map so more codes can be
# added later; an unknown current language toggles to Portuguese.
_LANG_TOGGLE = {"en": "pt", "pt": "en"}


class Daemon:
    """Idle<->recording state machine. Synchronous and dependency-injected so the
    whole machine exercises with fakes and a fake clock — no hardware, no threads.

    `serve()` is the only thing that touches the socket; this class just decides
    what a `start`/`stop`/timeout means.
    """

    def __init__(
        self,
        recorder: _Capturer,
        engine: _Transcriber,
        inject: Callable[[str], None],
        *,
        notify: _Notifier | None = None,
        language: str = "en",
        max_record_s: float = DEFAULT_MAX_RECORD_S,
        idle_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._inject = inject
        self._notify = notify or NullNotifier()
        self._language = language
        self._max_record_s = max_record_s
        # A managed engine can be idle-unloaded and preloaded; a plain one can't.
        self._managed = isinstance(engine, _ManagedEngine)
        # Idle-unload needs a managed (unloadable) engine; disable it otherwise.
        self._idle_timeout_s = idle_timeout_s if self._managed else None
        self._clock = clock
        self._log = log
        self._recording = False
        self._start_t = 0.0
        self._last_use = clock()  # for the idle-unload timer

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, code: str) -> str:
        """Switch the transcription language and confirm it with a toast.

        `code` is a language code (`en`/`pt`/…) or `"toggle"` to flip en<->pt.
        The daemon owns the canonical language (drives the toggle + the recording
        toast) and pushes it to the engine — no model reload. Returns the code now
        active."""
        if code == "toggle":
            code = _LANG_TOGGLE.get(self._language, "pt")
        self._language = code
        self._engine.set_language(code)
        self._notify.language(code)
        return code

    def start(self) -> None:
        """Begin capturing. A `start` while already recording (a duplicate press,
        or a press after a missed release) restarts cleanly — it discards the
        orphaned capture and opens a fresh one rather than wedging or raising."""
        if self._recording:
            self._log("start while recording — restarting capture")
            self._recording = False
            try:
                self._recorder.stop()  # drop the orphaned capture
            except Exception as exc:  # noqa: BLE001 (best-effort teardown)
                self._log(f"discarding orphaned capture failed: {exc}")
        self._recorder.start()
        self._recording = True
        self._start_t = self._clock()
        self._last_use = self._start_t  # activity: reset the idle-unload timer
        # persistent "armed" toast (shows the active language) until stop replaces it
        self._notify.recording(self._language)
        # Capture is already live above; now rebuild the model (if idle-unloaded)
        # so the load overlaps the hold instead of landing on the release.
        self._preload()

    def _preload(self) -> None:
        """Best-effort model rebuild on the press. If the engine was idle-unloaded,
        build it now so the multi-second load overlaps the user speaking; a plain
        (non-managed) or already-loaded engine is a no-op. Never raises — a failed
        preload just leaves the release-path `transcribe` to reload and surface the
        error, so a press is never worse than before this optimization existed."""
        if not self._managed or self._engine.loaded:  # type: ignore[attr-defined]  # guarded: managed engine
            return
        try:
            self._engine.load()  # type: ignore[attr-defined]  # guarded: managed engine
        except Exception as exc:  # noqa: BLE001 (preload is a pure optimization)
            self._log(f"preload on press failed ({exc}) — will reload on release")

    def stop(self) -> str:
        """End capture, transcribe -> format -> inject, and return the emitted
        text. A `stop` with no active recording is a harmless no-op (a release
        that arrived after a safety auto-stop, or with no matching press)."""
        if not self._recording:
            self._log("stop with no active recording — ignored")
            return ""
        # Flip to idle *before* the fallible work so a recorder/engine error can't
        # leave us stuck "recording" — a failed utterance still returns to idle.
        self._recording = False
        self._last_use = self._clock()  # activity: restart the idle-unload countdown
        text = ""
        try:
            audio = self._recorder.stop()
            text = self._engine.transcribe(audio)
            if text:
                self._inject(text)
            return text
        finally:
            # Always replace the persistent recording toast, even if transcribe/
            # inject raised — otherwise the -t 0 toast hangs on screen forever.
            self._notify.done(text)

    def remaining(self) -> float | None:
        """Seconds left before the safety auto-stop, or None when idle. `serve()`
        uses this as its accept() timeout so the loop wakes exactly in time."""
        if not self._recording:
            return None
        return max(0.0, self._max_record_s - (self._clock() - self._start_t))

    def check_timeout(self) -> str | None:
        """Safety auto-stop: if the max record duration elapsed, stop as if a
        `stop` arrived (returns the emitted text); else None. This is the
        anti-wedge backstop for a dropped release."""
        if self._recording and self._clock() - self._start_t >= self._max_record_s:
            self._log(f"safety auto-stop after {self._max_record_s:.0f}s (missed release?)")
            return self.stop()
        return None

    def idle_remaining(self) -> float | None:
        """Seconds until the idle-unload fires, or None when it can't/shouldn't —
        idle-unload disabled, currently recording, or the model already unloaded.
        `serve()` folds this into its accept() timeout so it wakes in time to drop
        the model."""
        if self._idle_timeout_s is None or self._recording:
            return None
        if not self._engine.loaded:  # type: ignore[attr-defined]  # guarded: managed engine
            return None
        return max(0.0, self._idle_timeout_s - (self._clock() - self._last_use))

    def check_idle(self) -> bool:
        """Drop the warm model if it's been idle past the timeout, freeing VRAM.
        Returns True iff it unloaded. No-op while recording, when disabled, or when
        the model is already unloaded."""
        if self._idle_timeout_s is None or self._recording:
            return False
        engine = self._engine
        if not engine.loaded:  # type: ignore[attr-defined]  # guarded: managed engine
            return False
        if self._clock() - self._last_use >= self._idle_timeout_s:
            self._log(f"idle {self._idle_timeout_s:.0f}s — unloading model to free VRAM")
            engine.unload()  # type: ignore[attr-defined]  # guarded: managed engine
            return True
        return False

    def abort(self) -> None:
        """Drop any in-progress capture without transcribing (shutdown path)."""
        if self._recording:
            self._recording = False
            try:
                self._recorder.stop()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown)
                self._log(f"abort: recorder stop failed: {exc}")


def _dispatch(daemon: Daemon, cmd: str) -> None:
    parts = cmd.split()
    if not parts:
        return
    verb = parts[0]
    if verb == "start":
        daemon.start()
    elif verb == "stop":
        daemon.stop()
    elif verb == "lang":
        # `lang <code>` sets it explicitly; bare `lang` flips en<->pt.
        daemon.set_language(parts[1] if len(parts) > 1 else "toggle")
    else:
        print(f"susurro: unknown command {cmd!r}", file=sys.stderr, flush=True)


def serve(daemon: Daemon, sock_path: str | None = None) -> int:
    """Socket shell around a `Daemon`: bind the Unix socket, dispatch start/stop,
    and wake on the safety-timeout deadline to run the auto-stop.

    Single client, single-flight: transcription blocks the accept loop, which is
    the intended serialization (one utterance at a time). Single-threaded — the
    safety timeout rides accept()'s socket timeout, so there are no locks.
    """
    sock_path = sock_path or socket_path()
    if os.path.exists(sock_path):
        os.unlink(sock_path)  # clear a stale socket from a crashed prior daemon
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen()
    print(f"susurro daemon: listening on {sock_path}", flush=True)
    try:
        while True:
            # Wake by the nearest of two deadlines: the recording auto-stop and the
            # idle-unload. Either may be None (not armed); None -> block until a
            # command. Floor a tiny positive value so accept() stays in timeout mode
            # (0.0 would flip the socket to non-blocking and busy-spin).
            deadlines = [d for d in (daemon.remaining(), daemon.idle_remaining()) if d is not None]
            srv.settimeout(None if not deadlines else max(min(deadlines), 0.05))
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                daemon.check_timeout()  # safety window elapsed with no stop
                daemon.check_idle()  # idle window elapsed -> free VRAM
                continue
            with conn:
                cmd = conn.recv(64).decode(errors="replace").strip()
            try:
                _dispatch(daemon, cmd)
            except Exception as exc:  # noqa: BLE001 (keep the daemon alive)
                print(f"susurro: command {cmd!r} failed: {exc}", file=sys.stderr, flush=True)
                daemon.abort()  # never leave the mic stuck open on an error
    except KeyboardInterrupt:
        print("\nsusurro daemon: bye", flush=True)
    finally:
        daemon.abort()  # release the mic if we were mid-capture at shutdown
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
    return 0


def _parse_device(value: str) -> int | str:
    """A numeric `--device` is a PortAudio index; anything else is a name substring."""
    return int(value) if value.isdigit() else value


def _build_parser() -> argparse.ArgumentParser:
    # Flag defaults are None ("not passed") so `_apply_cli` only overrides the config
    # value when a flag is actually given: built-in defaults < config file < CLI flag.
    p = argparse.ArgumentParser(prog="susurro-daemon", description=__doc__)
    p.add_argument("--config", default=None, help="path to config.toml (default: the repo-local config.toml)")
    p.add_argument("--model", default=None, help="faster-whisper model name")
    p.add_argument("--device", type=_parse_device, default=None, help="input device index or name")
    p.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    p.add_argument(
        "--lang",
        "--language",
        dest="lang",
        default=None,
        help="startup transcription language code (e.g. en, pt); live-switch with susurro-ctl lang",
    )
    p.add_argument(
        "--max-record",
        type=float,
        default=None,
        help="safety auto-stop after this many seconds",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="unload the model to free VRAM after this many idle seconds (<=0 disables)",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="disable the recording/done desktop notifications",
    )
    return p


def _apply_cli(config: Config, args: argparse.Namespace) -> Config:
    """Layer CLI flags over the loaded config (defaults < file < flag).

    `--cpu` and `--no-notify` are force-off switches (there's no matching on flag),
    so they override the config only in the off direction."""
    engine = replace(
        config.engine,
        model=pick(args.model, config.engine.model),
        device="cpu" if args.cpu else config.engine.device,
        language=pick(args.lang, config.engine.language),
    )
    audio = replace(config.audio, device=pick(args.device, config.audio.device))
    daemon = replace(
        config.daemon,
        max_record_s=pick(args.max_record, config.daemon.max_record_s),
        idle_timeout_s=pick(args.idle_timeout, config.daemon.idle_timeout_s),
        notify=config.daemon.notify and not args.no_notify,
    )
    return replace(config, engine=engine, audio=audio, daemon=daemon)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = _apply_cli(load_config(args.config), args)
    except ConfigError as exc:
        print(f"susurro: {exc}", file=sys.stderr, flush=True)
        return 1

    eng, aud, dae = config.engine, config.audio, config.daemon
    idle_timeout = dae.idle_timeout_s if dae.idle_timeout_s > 0 else None

    print(f"susurro daemon: loading {eng.model} on {eng.device} ({eng.language}) ...", flush=True)
    # LazyEngine so an idle daemon can drop the model and free VRAM, reloading via
    # this factory on the next utterance.
    engine = LazyEngine(
        lambda: Engine(
            eng.model,
            device=eng.device,
            compute_type=eng.compute_type,
            language=eng.language,
            beam_size=eng.beam_size,
            vad_filter=eng.vad_filter,
        ),
        language=eng.language,
    )
    # Warm now so the first real utterance already hits warm timing (this also
    # triggers the initial load); after an idle-unload the reload is lazy.
    engine.transcribe(np.zeros(aud.sample_rate // 2, dtype=np.float32))

    # Give the recorder headroom over the daemon timeout so the daemon's auto-stop
    # is always the authoritative stop; the recorder cap is a pure memory backstop.
    recorder = Recorder(
        max_duration_s=dae.max_record_s + 5.0,
        samplerate=aud.sample_rate,
        channels=aud.channels,
        device=aud.device,
    )
    notifier = Notifier(timeout_s=config.notify.timeout_s) if dae.notify else NullNotifier()
    daemon = Daemon(
        recorder,
        engine,
        inject,
        notify=notifier,
        language=eng.language,
        max_record_s=dae.max_record_s,
        idle_timeout_s=idle_timeout,
    )
    if idle_timeout:
        print(f"susurro daemon: idle-unload after {idle_timeout:.0f}s", flush=True)
    print("susurro daemon: ready (warm).", flush=True)
    return serve(daemon)


if __name__ == "__main__":
    sys.exit(main())
