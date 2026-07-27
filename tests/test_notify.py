"""Unit tests for the notification seam (`susurro.notify`).

`notify-send` is external and needs a running notification daemon, so
`subprocess.run` is patched. We assert the command shape (app name + the mako
sync hint that makes the two toasts share one slot), the persistent-vs-fading
timeouts, the empty-transcript fallback, the two failure toasts (wtype didn't type
it; the language code was refused), and the never-raise contract (a missing binary
or a hung spawn is swallowed, never surfaced to the daemon loop).
"""

import subprocess
from unittest import mock

from susurro.notify import NullNotifier, Notifier


def _cmd(run):
    (args,), _kwargs = run.call_args
    return args


def test_recording_toast_is_persistent_and_tagged():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().recording("en")
    cmd = _cmd(run)
    assert cmd[0] == "notify-send"
    assert cmd[-1] == "🎙 Recording (en)…"  # shows the active language
    assert ["-t", "0"] == cmd[cmd.index("-t") : cmd.index("-t") + 2]  # never expires
    assert "-a" in cmd and "susurro" in cmd
    assert "string:x-canonical-private-synchronous:susurro" in cmd  # shared slot


def test_recording_toast_shows_the_given_language():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().recording("pt")
    assert _cmd(run)[-1] == "🎙 Recording (pt)…"


def test_language_toast_uses_friendly_name_and_fades():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().language("pt")
    cmd = _cmd(run)
    assert cmd[-1] == "🌐 Susurro: Transcribing to Português"  # friendly display name
    assert cmd[cmd.index("-t") + 1] != "0"  # transient -> fades on its own
    assert "string:x-canonical-private-synchronous:susurro" in cmd


def test_language_toast_falls_back_to_raw_code():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().language("fr")  # no friendly name mapped
    assert _cmd(run)[-1] == "🌐 Susurro: Transcribing to fr"


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


def test_done_reports_a_failed_injection_instead_of_success():
    # wtype couldn't type it: the toast must not read "✓ Done". The transcript stays
    # in the body so the words are still recoverable by eye.
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().done("hello world", injected=False)
    cmd = _cmd(run)
    assert "⚠ Not typed (wtype failed)" in cmd
    assert "✓ Done" not in cmd
    assert cmd[-1] == "hello world"  # body keeps the transcript
    assert cmd[cmd.index("-u") + 1] == "normal"  # louder than the success toast
    assert "string:x-canonical-private-synchronous:susurro" in cmd  # still replaces


def test_done_defaults_to_the_success_toast():
    # The flag defaults to True, so a caller that doesn't pass it can't claim failure.
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().done("hi")
    assert "✓ Done" in _cmd(run)


def test_language_toast_reports_a_rejected_code():
    # An unsupported code changes nothing, so the toast names the offending code
    # rather than confirming a switch that didn't happen.
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().language("xx", supported=False)
    cmd = _cmd(run)
    assert cmd[-1] == '⚠ Susurro: unknown language "xx" — unchanged'
    assert cmd[cmd.index("-u") + 1] == "normal"
    assert cmd[cmd.index("-t") + 1] != "0"  # transient -> fades on its own


def test_missing_notify_send_is_swallowed(capsys):
    with mock.patch("susurro.notify.subprocess.run", side_effect=FileNotFoundError):
        Notifier().recording("en")  # must not raise
    assert "notify-send failed" in capsys.readouterr().err


def test_hung_notify_send_is_swallowed(capsys):
    with mock.patch(
        "susurro.notify.subprocess.run",
        side_effect=subprocess.TimeoutExpired("notify-send", 5.0),
    ):
        Notifier().done("hi")  # a stuck spawn must not wedge the daemon
    assert "notify-send failed" in capsys.readouterr().err


def test_notifier_forwards_configured_subprocess_timeout():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier(timeout_s=12.0).recording("en")
    assert run.call_args.kwargs["timeout"] == 12.0


def test_notifier_defaults_subprocess_timeout():
    with mock.patch("susurro.notify.subprocess.run") as run:
        Notifier().done("hi")
    assert run.call_args.kwargs["timeout"] == 5.0


def test_null_notifier_is_silent():
    # Also pins the signatures: the null notifier has to accept the outcome flags,
    # or `--no-notify` would blow up on exactly the failure paths that added them.
    with mock.patch("susurro.notify.subprocess.run") as run:
        NullNotifier().recording("en")
        NullNotifier().done("hi")
        NullNotifier().done("hi", injected=False)
        NullNotifier().language("pt")
        NullNotifier().language("xx", supported=False)
    run.assert_not_called()
