"""Unit tests for the socket shell (`susurro.daemon._serve_once` / `serve`).

The state machine is covered in `test_daemon.py`; this locks the loop body that
touches the socket — the part `test_daemon.py` can't reach. A real `AF_UNIX`
listener stands in for the daemon socket and a `FakeDaemon` records the calls the
loop makes, so we assert: a command is dispatched, an accept() timeout runs the
safety/idle checks, a silent client is dropped (not wedged), a dispatch error
triggers abort(), the accept() timeout is clamped (an unusable deadline must not
kill the loop), and `serve()` cleans up its socket on shutdown.
"""

import socket

import pytest

from susurro import daemon as daemon_mod
from susurro.daemon import _MAX_ACCEPT_TIMEOUT_S, _accept_timeout, _serve_once, serve


class FakeDaemon:
    """Records every loop-visible call so tests can assert the socket shell's
    behaviour without a real engine/recorder."""

    def __init__(self, remaining=None, idle_remaining=None):
        self._remaining = remaining
        self._idle_remaining = idle_remaining
        self.events = []

    def remaining(self):
        return self._remaining

    def idle_remaining(self):
        return self._idle_remaining

    def check_timeout(self):
        self.events.append("check_timeout")
        return None

    def check_idle(self):
        self.events.append("check_idle")
        return False

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")
        return ""

    def set_language(self, code):
        self.events.append(("lang", code))
        return code

    def abort(self):
        self.events.append("abort")


class BoomDaemon(FakeDaemon):
    def start(self):
        raise RuntimeError("boom")


class BoomTimeoutDaemon(FakeDaemon):
    """The auto-stop transcribe blows up: check_timeout raises like a real
    recorder/engine failure would on the safety-window path."""

    def check_timeout(self):
        raise RuntimeError("boom")


def _server(tmp_path) -> socket.socket:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(tmp_path / "susurro.sock"))
    srv.listen()
    return srv


def _connect(tmp_path) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(tmp_path / "susurro.sock"))
    return client


# --- _serve_once -----------------------------------------------------------


def test_serve_once_dispatches_a_command(tmp_path):
    srv = _server(tmp_path)
    client = _connect(tmp_path)
    client.sendall(b"start")
    daemon = FakeDaemon()
    try:
        _serve_once(srv, daemon)  # a connection is pending -> accept returns at once
    finally:
        client.close()
        srv.close()
    assert daemon.events == ["start"]


def test_serve_once_timeout_runs_safety_and_idle_checks(tmp_path):
    # No client connects; the nearest deadline (0.05s here) fires accept()'s timeout,
    # which must run both the safety auto-stop and the idle-unload checks.
    srv = _server(tmp_path)
    daemon = FakeDaemon(remaining=0.05)
    try:
        _serve_once(srv, daemon)
    finally:
        srv.close()
    assert daemon.events == ["check_timeout", "check_idle"]


def test_serve_once_survives_a_raising_auto_stop(tmp_path, capsys):
    # A failing auto-stop (transcribe/recorder error on the safety-window path) must
    # be caught and aborted, exactly like a failing command — never crash the loop.
    srv = _server(tmp_path)
    daemon = BoomTimeoutDaemon(remaining=0.05)  # no client -> the 0.05s deadline fires
    try:
        _serve_once(srv, daemon)
    finally:
        srv.close()
    assert daemon.events == ["abort"]
    assert "failed" in capsys.readouterr().err


def test_serve_once_drops_a_silent_client(tmp_path, monkeypatch, capsys):
    # A client that connects but never sends must be dropped after the recv timeout,
    # not block the loop forever. Shrink the timeout so the test is fast.
    monkeypatch.setattr(daemon_mod, "_CLIENT_RECV_TIMEOUT_S", 0.05)
    srv = _server(tmp_path)
    client = _connect(tmp_path)  # connect, send nothing
    daemon = FakeDaemon()
    try:
        _serve_once(srv, daemon)
    finally:
        client.close()
        srv.close()
    assert daemon.events == []  # nothing dispatched, no wedge
    assert "stalled" in capsys.readouterr().err


def test_serve_once_aborts_on_dispatch_error(tmp_path, capsys):
    srv = _server(tmp_path)
    client = _connect(tmp_path)
    client.sendall(b"start")
    daemon = BoomDaemon()
    try:
        _serve_once(srv, daemon)  # start() raises -> caught, abort() called
    finally:
        client.close()
        srv.close()
    assert daemon.events == ["abort"]
    assert "failed" in capsys.readouterr().err


def test_serve_once_survives_a_non_finite_deadline(tmp_path):
    # A non-finite deadline (an inf max_record_s/idle_timeout_s slipping past the
    # config validators) used to reach settimeout(inf) -> OverflowError, thrown
    # *outside* the try that guards accept(), so it escaped serve() and killed the
    # daemon. It must be clamped instead, and the command still dispatched.
    srv = _server(tmp_path)
    client = _connect(tmp_path)
    client.sendall(b"start")
    daemon = FakeDaemon(remaining=float("inf"))
    try:
        _serve_once(srv, daemon)  # must not raise
    finally:
        client.close()
        srv.close()
    assert daemon.events == ["start"]


def test_serve_once_ignores_blank_command(tmp_path):
    srv = _server(tmp_path)
    client = _connect(tmp_path)
    client.sendall(b"   ")  # whitespace -> _dispatch no-op
    daemon = FakeDaemon()
    try:
        _serve_once(srv, daemon)
    finally:
        client.close()
        srv.close()
    assert daemon.events == []


# --- accept() timeout clamping --------------------------------------------


def test_accept_timeout_is_none_when_nothing_is_armed():
    assert _accept_timeout([]) is None  # no deadline -> block until a command


def test_accept_timeout_picks_the_nearest_deadline():
    assert _accept_timeout([12.0, 3.0]) == pytest.approx(3.0)


def test_accept_timeout_floors_zero_to_stay_in_timeout_mode():
    # 0.0 would flip the socket to non-blocking and busy-spin the loop.
    assert _accept_timeout([0.0]) > 0.0


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1e300])
def test_accept_timeout_clamps_unusable_deadlines(bad):
    # inf/nan (and any absurd finite value) must never reach settimeout(): every
    # comparison against nan is False, so the ceiling needs its own finiteness check.
    assert _accept_timeout([bad]) == _MAX_ACCEPT_TIMEOUT_S


# --- serve() setup / teardown ---------------------------------------------


def test_serve_clears_stale_socket_and_cleans_up(tmp_path, monkeypatch):
    # A stale socket file from a crashed prior daemon must be unlinked before bind,
    # and the socket path must be removed on shutdown. Break the forever loop by
    # making the (patched) loop body raise KeyboardInterrupt after one turn.
    sock = tmp_path / "susurro.sock"
    sock.write_text("stale")  # leftover file at the bind path

    def fake_once(srv, daemon):
        raise KeyboardInterrupt

    monkeypatch.setattr(daemon_mod, "_serve_once", fake_once)
    daemon = FakeDaemon()
    rc = serve(daemon, str(sock))
    assert rc == 0
    assert daemon.events == ["abort"]  # teardown released any in-flight capture
    assert not sock.exists()  # socket cleaned up on exit


@pytest.mark.parametrize("cmd", [b"start", b"stop", b"lang pt", b"lang"])
def test_serve_once_routes_each_verb(tmp_path, cmd):
    srv = _server(tmp_path)
    client = _connect(tmp_path)
    client.sendall(cmd)
    daemon = FakeDaemon()
    try:
        _serve_once(srv, daemon)
    finally:
        client.close()
        srv.close()
    assert daemon.events  # every known verb reached the daemon
