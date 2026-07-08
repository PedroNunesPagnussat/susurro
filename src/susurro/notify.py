"""Desktop notifications for recording state, via `notify-send` (libnotify).

Hold-to-talk has no built-in "armed" indicator (a reason toggle was declined),
so the daemon posts a persistent "recording" toast while the key is held and
replaces it with the transcribed text on release. mako (the box's notification
daemon) collapses toasts that share the `x-canonical-private-synchronous` hint
into one slot, so the stop toast *replaces* the recording one instead of stacking
a second — one notification, two states.

Never raises: notifications are cosmetic, so a missing/failed/stuck `notify-send`
is swallowed (surfaced on stderr) rather than crashing the long-lived daemon or
wedging its single-threaded accept loop. The generous `subprocess` timeout is a
pathological-hang backstop, not a latency knob — notify-send returns immediately.
"""

from __future__ import annotations

import subprocess
import sys

# mako/dunst collapse toasts sharing this hint into one slot -> the stop toast
# replaces the persistent recording one rather than stacking a second.
_SYNC_HINT = "string:x-canonical-private-synchronous:susurro"
_APP = "susurro"
# Generous: notify-send should return at once; this only kills a stuck spawn so
# it can't wedge the accept loop. Not a latency knob.
_TIMEOUT_S = 5.0
# Friendly display names for the language toasts; unknown codes show the raw code.
_LANG_NAMES = {"en": "English", "pt": "Português"}


def _send(*args: str, timeout: float = _TIMEOUT_S) -> None:
    """Fire notify-send, swallowing every failure onto stderr. A missing binary,
    a non-zero return, or a hung spawn must never reach the daemon loop."""
    try:
        subprocess.run(
            ["notify-send", "-a", _APP, "-h", _SYNC_HINT, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"susurro: notify-send failed: {exc}", file=sys.stderr, flush=True)


class Notifier:
    """Default notifier: a persistent recording toast, replaced by the transcript.

    `timeout_s` caps the `notify-send` spawn (a hung one can't wedge the daemon's
    single-threaded loop); it comes from `[notify] timeout_s` in the config.
    """

    def __init__(self, timeout_s: float = _TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def recording(self, language: str) -> None:
        # -t 0 = never expire: it stays up for the whole hold as the armed
        # indicator, until `done()` replaces it via the shared sync hint. Shows
        # the active language so you can see what it'll transcribe as.
        _send("-t", "0", "-u", "low", f"🎙 Recording ({language})…", timeout=self._timeout_s)

    def done(self, text: str) -> None:
        # Replaces the persistent toast and fades on its own. Called on *every*
        # stop (even empty/failed), else the -t 0 toast would hang on screen.
        body = text.strip() or "(no speech)"
        _send("-t", "4000", "-u", "low", "✓ Done", body, timeout=self._timeout_s)

    def language(self, code: str) -> None:
        # Transient confirmation of a live language switch. Shares the sync slot,
        # so it's a quick standalone toast (you switch while not recording).
        name = _LANG_NAMES.get(code, code)
        _send(
            "-t", "2000", "-u", "low", f"🌐 Susurro: Transcribing to {name}",
            timeout=self._timeout_s,
        )


class NullNotifier:
    """No-op notifier for `--no-notify` and the daemon's default (opt-in toasts)."""

    def recording(self, language: str) -> None: ...
    def done(self, text: str) -> None: ...
    def language(self, code: str) -> None: ...
