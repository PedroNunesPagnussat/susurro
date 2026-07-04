"""Unit tests for the notification seam (`susurro.notify`).

`notify-send` is external and needs a running notification daemon, so
`subprocess.run` is patched. We assert the command shape (app name + the mako
sync hint that makes the two toasts share one slot), the persistent-vs-fading
timeouts, the empty-transcript fallback, and the never-raise contract (a missing
binary or a hung spawn is swallowed, never surfaced to the daemon loop).
"""

import subprocess
from unittest import mock

from susurro.notify import NullNotifier, Notifier


def _cmd(run):
    (args,), _kwargs = run.call_args
    return args


def test_recording_toast_is_persistent_and_tagged():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().recording()
    cmd = _cmd(run)
    assert cmd[0] == "notify-send"
    assert cmd[-1] == "🎙 Recording…"
    assert ["-t", "0"] == cmd[cmd.index("-t") : cmd.index("-t") + 2]  # never expires
    assert "-a" in cmd and "susurro" in cmd
    assert "string:x-canonical-private-synchronous:susurro" in cmd  # shared slot


def test_done_toast_shows_transcript_and_fades():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().done("hello world")
    cmd = _cmd(run)
    assert cmd[-1] == "hello world"  # body is the transcript
    assert cmd[cmd.index("-t") + 1] != "0"  # finite timeout -> fades on its own
    assert "string:x-canonical-private-synchronous:susurro" in cmd


def test_done_falls_back_when_transcript_empty():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().done("   ")
    assert _cmd(run)[-1] == "(no speech)"


def test_missing_notify_send_is_swallowed(capsys):
    with mock.patch("susurro.notify.subprocess.run", side_effect=FileNotFoundError):
        Notifier().recording()  # must not raise
    assert "notify-send failed" in capsys.readouterr().err


def test_hung_notify_send_is_swallowed(capsys):
    with mock.patch(
        "susurro.notify.subprocess.run",
        side_effect=subprocess.TimeoutExpired("notify-send", 5.0),
    ):
        Notifier().done("hi")  # a stuck spawn must not wedge the daemon
    assert "notify-send failed" in capsys.readouterr().err


def test_null_notifier_is_silent():
    with mock.patch("susurro.notify.subprocess.run") as run:
        NullNotifier().recording()
        NullNotifier().done("hi")
    run.assert_not_called()
