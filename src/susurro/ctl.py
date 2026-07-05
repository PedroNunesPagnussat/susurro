"""Thin hold-to-talk client: `susurro-ctl {start,stop}` -> one Unix-socket write.

Bound to a Hyprland key: `bind` (press) -> `start`, `bindr` (release) -> `stop`
(see README). Deliberately stdlib-only and tiny (no numpy, no engine) so a fresh
process per key press stays cheap; the daemon owns all the cost, and release->text
latency stays inference-bound.

Exit codes: 0 sent, 1 daemon not running, 2 bad usage.
"""

from __future__ import annotations

import socket
import sys

from ._ipc import socket_path


def send(cmd: str) -> int:
    """Open the daemon socket, write `cmd`, close. Returns 0 on success, 1 if the
    daemon isn't running (socket missing or not accepting)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(socket_path())
            s.sendall(cmd.encode())
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        print(f"susurro-ctl: daemon not running ({exc})", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in {"start", "stop"}:
        print("usage: susurro-ctl {start|stop}", file=sys.stderr)
        return 2
    return send(argv[0])


if __name__ == "__main__":
    sys.exit(main())
