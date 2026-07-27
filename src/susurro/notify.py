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

Cosmetic, but not optional: the daemon is autostarted from Hyprland's `exec-once`,
so stderr goes nowhere and these toasts are the *only* channel to the user. Both
`done` and `language` therefore take a keyword-only outcome flag so a failure
(wtype didn't type it; the language code was rejected) is reported here instead of
being dressed up as success. The flags default to the success value so a caller
that doesn't know about them can't accidentally claim a failure.
"""

from __future__ import annotations

import subprocess
import sys

# mako/dunst collapse toasts sharing this hint into one slot -> the stop toast
# replaces the persistent recording one rather than stacking a second.
_SYNC_HINT = "string:x-canonical-private-synchronous:susurro"
_APP = "susurro"
# Generous: notify-send should return at once; this only kills a stuck spawn so
# it can't wedge the accept loop. Not a latency knob. Canonical default that also
# backs `NotifyConfig.timeout_s` (single source of truth).
DEFAULT_TIMEOUT_S = 5.0
# Friendly display names for the language toasts; unknown codes show the raw code.
_LANG_NAMES = {"en": "English", "pt": "Português"}


def _send(*args: str, timeout: float = DEFAULT_TIMEOUT_S) -> None:
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

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def recording(self, language: str) -> None:
        # -t 0 = never expire: it stays up for the whole hold as the armed
        # indicator, until `done()` replaces it via the shared sync hint. Shows
        # the active language so you can see what it'll transcribe as.
        _send("-t", "0", "-u", "low", f"🎙 Recording ({language})…", timeout=self._timeout_s)

    def done(self, text: str, *, injected: bool = True) -> None:
        # Replaces the persistent toast and fades on its own. Called on *every*
        # stop (even empty/failed), else the -t 0 toast would hang on screen.
        body = text.strip() or "(no speech)"
        if not injected:
            # wtype couldn't type it (missing binary, compositor refusal). Say so
            # instead of "✓ Done": normal urgency and a longer dwell so it's noticed,
            # and the transcript stays in the body so the words aren't simply lost.
            _send(
                "-t",
                "8000",
                "-u",
                "normal",
                "⚠ Not typed (wtype failed)",
                body,
                timeout=self._timeout_s,
            )
            return
        _send("-t", "4000", "-u", "low", "✓ Done", body, timeout=self._timeout_s)

    def language(self, code: str, *, supported: bool = True) -> None:
        # Transient confirmation of a live language switch. Shares the sync slot,
        # so it's a quick standalone toast (you switch while not recording).
        if not supported:
            # The switch was refused, so nothing changed. Name the offending code:
            # the alternative was a confirming toast followed by every later
            # utterance silently failing inside the model.
            _send(
                "-t",
                "5000",
                "-u",
                "normal",
                f'⚠ Susurro: unknown language "{code}" — unchanged',
                timeout=self._timeout_s,
            )
            return
        name = _LANG_NAMES.get(code, code)
        _send(
            "-t",
            "2000",
            "-u",
            "low",
            f"🌐 Susurro: Transcribing to {name}",
            timeout=self._timeout_s,
        )


class NullNotifier:
    """No-op notifier for `--no-notify` and the daemon's default (opt-in toasts)."""

    def recording(self, language: str) -> None: ...
    def done(self, text: str, *, injected: bool = True) -> None: ...
    def language(self, code: str, *, supported: bool = True) -> None: ...
