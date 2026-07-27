"""Desktop notifications for recording state, via `notify-send` (libnotify).

Hold-to-talk has no built-in "armed" indicator (a reason toggle was declined),
so the daemon posts a persistent "recording" toast while the key is held and
replaces it with the transcribed text on release. mako (the box's notification
daemon) collapses toasts that share the `x-canonical-private-synchronous` hint
into one slot, so the stop toast *replaces* the recording one instead of stacking
a second — one notification, two states.

Never raises: notifications are cosmetic, so a missing/failed/stuck `notify-send`
is swallowed (surfaced on stderr) rather than crashing the long-lived daemon or
wedging its single-threaded accept loop.

Cosmetic, but not optional: the daemon is autostarted from Hyprland's `exec-once`,
so stderr goes nowhere and these toasts are the *only* channel to the user. Both
`done` and `language` therefore take a keyword-only outcome flag so a failure
(wtype didn't type it; the language code was rejected) is reported here instead of
being dressed up as success, defaulting to the success value.
"""

from __future__ import annotations

import subprocess
import sys

_SYNC_HINT = "string:x-canonical-private-synchronous:susurro"
_APP = "susurro"
# Generous: only kills a stuck spawn so it can't wedge the accept loop, not a
# latency knob. Also backs `NotifyConfig.timeout_s` (single source of truth).
DEFAULT_TIMEOUT_S = 5.0
# Friendly display names for the language toasts; unknown codes show the raw code.
_LANG_NAMES = {"en": "English", "pt": "Português"}


def _send(
    summary: str,
    body: str | None = None,
    *,
    expire_ms: int,
    urgency: str,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> None:
    """Fire notify-send, swallowing every failure onto stderr. A missing binary,
    a non-zero return, or a hung spawn must never reach the daemon loop.

    Summary and body go after `--` because the body is *user text*: GLib's option
    parser reads a leading `-` as an unknown option and exits 1 without posting
    anything (`notify-send … "✓ Done" "-5 degrees"` -> `Unknown option -5 degrees`).
    Whisper emits dialogue dashes routinely in pt, so that is a real transcript,
    and a toast that never posts also never *replaces* the persistent `-t 0`
    recording one — it would hang on screen forever (same reason `inject.py` does it).

    A non-zero return is logged rather than ignored: it means the toast the daemon
    believes it posted isn't on screen."""
    args = ["notify-send", "-a", _APP, "-h", _SYNC_HINT, "-t", str(expire_ms), "-u", urgency]
    args += ["--", summary] if body is None else ["--", summary, body]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"susurro: notify-send failed: {exc}", file=sys.stderr, flush=True)
        return
    if proc.returncode != 0:
        print(
            f"susurro: notify-send failed (rc={proc.returncode}): {proc.stderr.strip()}",
            file=sys.stderr,
            flush=True,
        )


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
        _send(f"🎙 Recording ({language})…", expire_ms=0, urgency="low", timeout=self._timeout_s)

    def done(self, text: str, *, injected: bool = True) -> None:
        # Replaces the persistent toast and fades on its own. Called on *every*
        # stop (even empty/failed), else the -t 0 toast would hang on screen.
        body = text.strip() or "(no speech)"
        if not injected:
            # wtype couldn't type it. Longer dwell so it's noticed, and the transcript
            # stays in the body so the words aren't simply lost.
            _send(
                "⚠ Not typed (wtype failed)",
                body,
                expire_ms=8000,
                urgency="normal",
                timeout=self._timeout_s,
            )
            return
        _send("✓ Done", body, expire_ms=4000, urgency="low", timeout=self._timeout_s)

    def language(self, code: str, *, supported: bool = True) -> None:
        # Transient confirmation of a live language switch. Shares the sync slot,
        # so it's a quick standalone toast (you switch while not recording).
        if not supported:
            # Refused, so nothing changed. Name the code: the alternative was a
            # confirming toast, then every later utterance silently failing.
            _send(
                f'⚠ Susurro: unknown language "{code}" — unchanged',
                expire_ms=5000,
                urgency="normal",
                timeout=self._timeout_s,
            )
            return
        name = _LANG_NAMES.get(code, code)
        _send(
            f"🌐 Susurro: Transcribing to {name}",
            expire_ms=2000,
            urgency="low",
            timeout=self._timeout_s,
        )


class NullNotifier:
    """No-op notifier for `--no-notify` and the daemon's default (opt-in toasts)."""

    def recording(self, language: str) -> None: ...
    def done(self, text: str, *, injected: bool = True) -> None: ...
    def language(self, code: str, *, supported: bool = True) -> None: ...
