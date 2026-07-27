"""Text injection into the focused Wayland window via `wtype`.

The daemon's terminal step: on `stop`, the formatted transcript is typed into
whatever window holds focus. `wtype` speaks the Wayland virtual-keyboard protocol
(Hyprland-native, no uinput / no root / no `input` group).

Never raises: a non-zero return, a missing `wtype` binary, or a hung spawn are all
surfaced on stderr so an injection failure can't crash — or wedge — the long-lived,
single-threaded daemon. It *reports* the failure instead, via a bool return: stderr
goes nowhere under Hyprland's `exec-once`, so the caller needs a value it can turn
into something the user actually sees (the daemon turns it into the stop toast).
Without it a missing `wtype` — the most likely first-run failure — looked exactly
like success. The subprocess `timeout` is the anti-wedge backstop: `inject` runs
inside the daemon's `serve()` accept loop, so a stuck `wtype` (compositor stall)
must not block it forever (mirrors `notify.py`). Empty/whitespace text is a no-op (the
formatter already drops silence; this is the belt-and-braces guard so we never spawn
wtype with nothing to type).

For very long paragraphs or apps that drop fast synthetic keystrokes, clipboard +
paste (`wl-copy` + a synthesized paste) is more robust, but it clobbers the
clipboard and the paste shortcut is per-app, so `wtype` stays the default (see
README for the manual fallback).
"""

from __future__ import annotations

import subprocess
import sys

# Generous: wtype returns as fast as it can type the text; this only kills a spawn
# stuck on a wedged compositor so it can't block the daemon's accept loop. Long
# enough not to truncate a legitimately long paragraph. Not a latency knob.
_TIMEOUT_S = 30.0


def inject(text: str, *, timeout_s: float = _TIMEOUT_S) -> bool:
    """Type `text` into the focused window via wtype. No-op on empty/whitespace.

    Returns True iff the text actually landed in the focused window, False if wtype
    couldn't type it. Never raises: a non-zero wtype return, a missing binary, or a
    spawn that overruns `timeout_s` is surfaced on stderr *and* reported as False,
    instead of crashing or wedging the daemon.

    Empty text returns True: there was nothing to type, so nothing failed — the
    caller must not read it as an injection failure (see `notify.Notifier.done`)."""
    if not text.strip():
        return True
    try:
        # `--` ends wtype's option parsing so a transcript starting with `-` is
        # typed literally, not misread as a flag (man wtype: `... -- [TEXT]...`).
        proc = subprocess.run(
            ["wtype", "--", text], capture_output=True, text=True, timeout=timeout_s
        )
    except (OSError, subprocess.SubprocessError) as exc:  # missing binary / hung spawn
        print(f"susurro: wtype failed: {exc}", file=sys.stderr, flush=True)
        return False
    if proc.returncode != 0:
        print(
            f"susurro: wtype failed (rc={proc.returncode}): {proc.stderr.strip()}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True
