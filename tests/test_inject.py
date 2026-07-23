"""Unit tests for the injection seam (`susurro.inject.inject`).

`wtype` is external and Wayland-only, so `subprocess.run` is patched. We assert
the command built (`["wtype", "--", text]`), the empty/whitespace no-op (no subprocess
call at all), and the never-crash-*or*-wedge-the-daemon contract: a non-zero
return, a missing binary, and a hung spawn (timeout) are all surfaced on stderr
without raising, and a spawn timeout is always passed.
"""

import subprocess
from unittest import mock

import pytest

from susurro.inject import inject


def test_inject_runs_wtype_with_text():
    with mock.patch("susurro.inject.subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr="")
        inject("hello world")
    run.assert_called_once()
    (cmd,), _kwargs = run.call_args
    # `--` ends option parsing so a leading-dash transcript can't be read as a flag.
    assert cmd == ["wtype", "--", "hello world"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_or_whitespace_text_is_noop(text):
    with mock.patch("susurro.inject.subprocess.run") as run:
        inject(text)
    run.assert_not_called()  # nothing to type -> no wtype spawn


def test_wtype_failure_is_surfaced_not_raised(capsys):
    with mock.patch("susurro.inject.subprocess.run") as run:
        run.return_value = mock.Mock(returncode=1, stderr="no compositor")
        inject("hi")  # must not raise
    err = capsys.readouterr().err
    assert "wtype failed" in err
    assert "no compositor" in err


def test_missing_wtype_binary_is_swallowed(capsys):
    # wtype not installed -> FileNotFoundError (an OSError). Must not raise: the
    # docstring's never-raises contract covers a missing binary, not just rc != 0.
    with mock.patch("susurro.inject.subprocess.run", side_effect=FileNotFoundError("wtype")):
        inject("hi")  # must not raise
    assert "wtype failed" in capsys.readouterr().err


def test_hung_wtype_is_swallowed(capsys):
    # A spawn stuck on a wedged compositor must time out, not block the daemon's
    # single-threaded accept loop forever.
    with mock.patch(
        "susurro.inject.subprocess.run",
        side_effect=subprocess.TimeoutExpired("wtype", 30.0),
    ):
        inject("hi")  # must not raise
    assert "wtype failed" in capsys.readouterr().err


def test_inject_passes_a_spawn_timeout():
    with mock.patch("susurro.inject.subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr="")
        inject("hi")
    assert run.call_args.kwargs["timeout"] > 0  # anti-wedge backstop always set
