"""Unit tests for the injection seam (`susurro.inject.inject`).

`wtype` is external and Wayland-only, so `subprocess.run` is patched. We assert
the three behaviours the plan named: the command built (`["wtype", text]`), the
empty/whitespace no-op (no subprocess call at all), and that a non-zero wtype
return is surfaced on stderr without raising (the never-crash-the-daemon contract).
"""

from unittest import mock

import pytest

from susurro.inject import inject


def test_inject_runs_wtype_with_text():
    with mock.patch("susurro.inject.subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr="")
        inject("hello world")
    run.assert_called_once()
    (cmd,), _kwargs = run.call_args
    assert cmd == ["wtype", "hello world"]


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
