#!/usr/bin/env python3
"""Phase 2 trigger + injection spike — NO audio, NO model, stdlib only.

Step-0-style de-risk before building the real daemon. Proves two unknowns:

  1. Hyprland `bind` (press) + `bindr` (release) drive clean start/stop over a
     Unix socket — i.e. one hold => exactly one START then one STOP, and the
     measured hold duration matches how long the key was physically down.
  2. `wtype` injects text into the *focused* Wayland window (virtual-keyboard
     protocol; Hyprland-native, no uinput / no root / no `input` group).

Usage
-----
Leave the daemon running in a terminal:

    python3 scripts/trigger_spike.py daemon

Then either drive it manually from a second terminal (proves wtype works, but
types into *that* terminal since it holds focus):

    python3 scripts/trigger_spike.py start
    python3 scripts/trigger_spike.py stop

...or wire the real trigger in ~/.config/hypr/hyprland.conf and `hyprctl reload`:

    # Modifier-free key is cleanest (see NOTES). `Menu` types nothing on its own:
    bind  = , Menu, exec, python3 ~/dev/susurro/scripts/trigger_spike.py start
    bindr = , Menu, exec, python3 ~/dev/susurro/scripts/trigger_spike.py stop

Now focus an editor or browser, hold the key, wait, release. On release the
daemon types a marker like `[susurro spike: held 1.23s] ` into that window.

NOTES / what to watch for (these are the Phase 2 risks this spike surfaces)
--------------------------------------------------------------------------
* Modifier still held during injection: if you bind `SUPER, D`, releasing D
  fires `bindr` while SUPER may still be physically down, so wtype's keystrokes
  can combine with SUPER into window-manager actions. A modifier-free key
  (`Menu`, a spare `F13`+, `Pause`) avoids this entirely — recommended.
* Modifier-release ordering: with `SUPER, D`, Hyprland only fires `bindr` if
  SUPER is still held when D is released. Release the main key *before* the
  modifier, or (better) go modifier-free.
* Latency: each client call is a fresh `python3` start (~30ms) + socket send.
  Note whether release->text feels instant; if not, the real client gets
  rewritten in something lighter or made resident.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "susurro-spike.sock")


def _inject(text: str) -> None:
    """Type `text` into the focused window via wtype. check=False: never crash
    the daemon if injection fails (surface the returncode instead)."""
    proc = subprocess.run(["wtype", text], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  wtype failed (rc={proc.returncode}): {proc.stderr.strip()}", flush=True)


def daemon() -> int:
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen()
    print(f"spike daemon listening on {SOCK}")
    print("hold your Hyprland key (or send start/stop from another terminal). Ctrl-C quits.")

    start_t: float | None = None
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                cmd = conn.recv(64).decode().strip()
            now = time.perf_counter()
            stamp = time.strftime("%H:%M:%S")

            if cmd == "start":
                if start_t is not None:
                    print(f"[{stamp}] START while already recording (missed a STOP?)", flush=True)
                start_t = now
                print(f"[{stamp}] START", flush=True)
            elif cmd == "stop":
                if start_t is None:
                    print(f"[{stamp}] STOP with no START (release fired without press?)", flush=True)
                    continue
                held = now - start_t
                start_t = None
                print(f"[{stamp}] STOP  held={held:.3f}s -> injecting", flush=True)
                _inject(f"[susurro spike: held {held:.2f}s] ")
            else:
                print(f"[{stamp}] unknown cmd: {cmd!r}", flush=True)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.close()
        if os.path.exists(SOCK):
            os.unlink(SOCK)
    return 0


def client(cmd: str) -> int:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCK)
        s.sendall(cmd.encode())
        s.close()
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        print(f"susurro spike: daemon not running ({exc})", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"daemon", "start", "stop"}:
        print(__doc__)
        return 2
    return daemon() if argv[1] == "daemon" else client(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
