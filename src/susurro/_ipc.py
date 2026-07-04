"""Shared daemon<->client IPC location. Stdlib-only (just `os`) so the thin
client (`susurro.ctl`) imports nothing heavy — process startup stays cheap.

The socket lives in `$XDG_RUNTIME_DIR` (per-user, tmpfs, cleaned on logout);
falls back to `/tmp` only if the runtime dir is somehow unset.
"""

from __future__ import annotations

import os


def socket_path() -> str:
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "susurro.sock")
