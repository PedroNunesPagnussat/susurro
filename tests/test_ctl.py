"""Unit tests for the thin client (`susurro.ctl`).

No daemon/model needed: a bare stdlib Unix socket stands in for the daemon and we
assert the client writes the right command. `XDG_RUNTIME_DIR` is pointed at a
tmp dir so the client's real socket-path logic (`_ipc.socket_path`) is exercised.
"""

import socket
import threading
from pathlib import Path

from susurro import ctl


def _listen(sock_path: Path) -> tuple[socket.socket, list[bytes], threading.Thread]:
    """Bind a Unix listener and accept exactly one command in a background thread.

    Returns the thread so the caller can join it before asserting (the accept only
    records after the client's write lands — joining removes the race).
    """
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen()
    received: list[bytes] = []

    def accept_one():
        conn, _ = srv.accept()
        with conn:
            received.append(conn.recv(64))

    thread = threading.Thread(target=accept_one, daemon=True)
    thread.start()
    return srv, received, thread


def test_send_delivers_command_to_the_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    srv, received, thread = _listen(tmp_path / "susurro.sock")
    try:
        rc = ctl.send("start")
        thread.join(timeout=2)
    finally:
        srv.close()
    assert rc == 0
    assert received == [b"start"]


def test_main_dispatches_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    srv, received, thread = _listen(tmp_path / "susurro.sock")
    try:
        rc = ctl.main(["stop"])
        thread.join(timeout=2)
    finally:
        srv.close()
    assert rc == 0
    assert received == [b"stop"]


def test_send_returns_1_when_daemon_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))  # no socket bound here
    assert ctl.send("start") == 1


def test_main_forwards_lang_with_code(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    srv, received, thread = _listen(tmp_path / "susurro.sock")
    try:
        rc = ctl.main(["lang", "pt"])
        thread.join(timeout=2)
    finally:
        srv.close()
    assert rc == 0
    assert received == [b"lang pt"]


def test_main_bare_lang_forwards_toggle(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    srv, received, thread = _listen(tmp_path / "susurro.sock")
    try:
        rc = ctl.main(["lang"])
        thread.join(timeout=2)
    finally:
        srv.close()
    assert rc == 0
    assert received == [b"lang toggle"]


def test_main_rejects_bad_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert ctl.main([]) == 2
    assert ctl.main(["frobnicate"]) == 2
    assert ctl.main(["start", "stop"]) == 2
    assert ctl.main(["lang", "pt", "en"]) == 2  # too many args
