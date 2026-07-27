"""Thin hold-to-talk client: `susurro-ctl {start,stop}` -> one Unix-socket write.

Bound to a Hyprland key: `bind` (press) -> `start`, `bindr` (release) -> `stop`
(see README). Deliberately stdlib-only and tiny (no numpy, no engine) so a fresh
process per key press stays cheap; the daemon owns all the cost, and release->text
latency stays inference-bound.

Exit codes: 0 sent, 1 daemon unreachable, 2 bad usage.
"""

from __future__ import annotations

import socket
import sys

from ._ipc import socket_path


def send(cmd: str) -> int:
    """Open the daemon socket, write `cmd`, close. Returns 0 on success, 1 if the
    daemon can't be reached (socket missing, not accepting, or otherwise unusable).

    Catches every `OSError`, not just the missing/refused pair: a bad runtime dir, a
    permission error, or a reset mid-write must exit 1 with a diagnostic, never dump
    a traceback on the key-press this runs from."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(socket_path())
            s.sendall(cmd.encode())
    except OSError as exc:  # missing socket, refused, reset, bad path, no permission…
        print(f"susurro-ctl: cannot reach daemon ({exc})", file=sys.stderr)
        return 1
    return 0


_USAGE = "usage: susurro-ctl {start|stop|lang [pt|en|toggle]}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1 and argv[0] in {"start", "stop"}:
        return send(argv[0])
    if argv and argv[0] == "lang" and len(argv) <= 2:
        # bare `lang` = toggle en<->pt; `lang <code>` sets it explicitly. The daemon
        # owns code validation: it rejects an unknown code, keeps the current
        # language, and says so in a toast — so this stays a dumb pass-through.
        return send(f"lang {argv[1]}" if len(argv) == 2 else "lang toggle")
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
