"""Phase-2 daemon: the warm, always-on half of hold-to-talk.

Holds the warm `Engine` + a `Recorder` and drives an idle<->recording state
machine over a Unix socket (tiny `start`/`stop` line protocol). The client
(Step 9, `susurro-ctl`) is a trivial socket write bound to a Hyprland key; all the
cost (model load, CUDA warm) is paid once here at `exec-once` autostart, so
per-utterance latency is inference-bound.

On `stop` (or the safety auto-stop timeout): transcribe -> format -> inject into
the focused window. The state machine (`Daemon`) is pure and DI'd — recorder,
engine, inject, and clock are all injectable — so it unit-tests with fakes: no
mic, model, socket, or wall-clock. `serve()` is the thin socket shell around it.

The safety timeout is the anti-wedge backstop the Step-6 spike proved we need: a
missed key release (a dropped `stop`) must not leave the daemon recording forever.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from ._ipc import socket_path
from .audio import SAMPLE_RATE, Recorder
from .engine import DEFAULT_MODEL, Engine

DEFAULT_MAX_RECORD_S = 30.0


class _Capturer(Protocol):
    """Structural type for the recorder seam (real: `audio.Recorder`)."""

    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...


class _Transcriber(Protocol):
    """Structural type for the engine seam (real: `engine.Engine`)."""

    def transcribe(self, audio: np.ndarray) -> str: ...


def _wtype_inject(text: str) -> None:
    """Type `text` into the focused window via wtype. Minimal inline default,
    extracted + hardened into `susurro.inject` at Step 10. Never raises: a wtype
    failure is surfaced on stderr instead of crashing the daemon."""
    proc = subprocess.run(["wtype", text], capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"susurro: wtype failed (rc={proc.returncode}): {proc.stderr.strip()}",
            file=sys.stderr,
            flush=True,
        )


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
        max_record_s: float = DEFAULT_MAX_RECORD_S,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._inject = inject
        self._max_record_s = max_record_s
        self._clock = clock
        self._log = log
        self._recording = False
        self._start_t = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

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
        audio = self._recorder.stop()
        text = self._engine.transcribe(audio)
        if text:
            self._inject(text)
        return text

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

    def abort(self) -> None:
        """Drop any in-progress capture without transcribing (shutdown path)."""
        if self._recording:
            self._recording = False
            try:
                self._recorder.stop()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown)
                self._log(f"abort: recorder stop failed: {exc}")


def _dispatch(daemon: Daemon, cmd: str) -> None:
    if cmd == "start":
        daemon.start()
    elif cmd == "stop":
        daemon.stop()
    elif cmd:
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
            rem = daemon.remaining()
            # None -> block until a command; else wake by the auto-stop deadline.
            # Floor a tiny positive value so accept() stays in timeout mode (0.0
            # would flip the socket to non-blocking and busy-spin).
            srv.settimeout(None if rem is None else max(rem, 0.05))
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                daemon.check_timeout()  # safety window elapsed with no stop
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="susurro-daemon", description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model name")
    p.add_argument("--device", default=None, help="input device index or name")
    p.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    p.add_argument(
        "--max-record",
        type=float,
        default=DEFAULT_MAX_RECORD_S,
        help="safety auto-stop after this many seconds",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device_arg = args.device
    if isinstance(device_arg, str) and device_arg.isdigit():
        device_arg = int(device_arg)

    device = "cpu" if args.cpu else "cuda"
    print(f"susurro daemon: loading {args.model} on {device} ...", flush=True)
    engine = Engine(args.model, device=device)
    # Warm the CUDA kernels so the first real utterance already hits warm timing.
    engine.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    # Give the recorder headroom over the daemon timeout so the daemon's auto-stop
    # is always the authoritative stop; the recorder cap is a pure memory backstop.
    recorder = Recorder(max_duration_s=args.max_record + 5.0, device=device_arg)
    daemon = Daemon(recorder, engine, _wtype_inject, max_record_s=args.max_record)
    print("susurro daemon: ready (warm).", flush=True)
    return serve(daemon)


if __name__ == "__main__":
    sys.exit(main())
