"""Text injection into the focused Wayland window via `wtype`.

The daemon's terminal step: on `stop`, the formatted transcript is typed into
whatever window holds focus. `wtype` speaks the Wayland virtual-keyboard protocol
(Hyprland-native, no uinput / no root / no `input` group).

Never raises on a wtype failure: a non-zero return is surfaced on stderr so a
transient injection error can't crash the long-lived daemon. Empty/whitespace text
is a no-op (the formatter already drops silence; this is the belt-and-braces guard
so we never spawn wtype with nothing to type).

For very long paragraphs or apps that drop fast synthetic keystrokes, clipboard +
paste (`wl-copy` + a synthesized paste) is more robust, but it clobbers the
clipboard and the paste shortcut is per-app, so `wtype` stays the default (see
README for the manual fallback).
"""

from __future__ import annotations

import subprocess
import sys


def inject(text: str) -> None:
    """Type `text` into the focused window via wtype. No-op on empty/whitespace.
    Never raises: a non-zero wtype return is surfaced on stderr instead of
    crashing the daemon."""
    if not text.strip():
        return
    proc = subprocess.run(["wtype", text], capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"susurro: wtype failed (rc={proc.returncode}): {proc.stderr.strip()}",
            file=sys.stderr,
            flush=True,
        )
