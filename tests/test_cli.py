"""Unit tests for the shared CLI plumbing (`susurro._cli`).

`apply_engine_audio` is exercised through the daemon's `_apply_cli` tests (see
tests/test_daemon.py); here we lock the two small helpers directly (device parsing,
the positive-float guard) and prove the guard actually rejects a non-positive or
non-finite flag at parse time — the gap where a `--max-record 0` (or `inf`) used to
slip past the config's fail-loud checks.
"""

import argparse

import pytest

from susurro import _cli
from susurro.__main__ import _build_parser as build_main_parser
from susurro.daemon import _build_parser as build_daemon_parser


@pytest.mark.parametrize(
    ("value", "expected"), [("4", 4), ("0", 0), ("hw:1", "hw:1"), ("USB mic", "USB mic")]
)
def test_parse_device_int_vs_name(value, expected):
    assert _cli.parse_device(value) == expected


def test_positive_float_accepts_positive():
    assert _cli.positive_float("2.5") == 2.5


@pytest.mark.parametrize("value", ["0", "-1", "2.5"])
def test_finite_float_accepts_any_finite_value(value):
    # Unlike `positive_float`, non-positive is legal here: `--idle-timeout 0` is the
    # documented way to keep the model resident.
    assert _cli.finite_float(value) == float(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_finite_float_rejects_non_finite(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _cli.finite_float(value)


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_positive_float_rejects_non_positive(value):
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        _cli.positive_float(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity", "1e400"])
def test_positive_float_rejects_non_finite(value):
    # `float()` happily returns nan/inf (and silently overflows `1e400` to inf), and
    # every `<= 0` comparison against them is False — so they need their own check.
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _cli.positive_float(value)


def test_daemon_parser_rejects_non_positive_max_record():
    # The config file rejects max_record_s <= 0; the CLI flag must too (else every
    # recording would auto-stop instantly). argparse exits non-zero on a bad value.
    with pytest.raises(SystemExit):
        build_daemon_parser().parse_args(["--max-record", "0"])


def test_daemon_parser_accepts_positive_max_record():
    args = build_daemon_parser().parse_args(["--max-record", "90"])
    assert args.max_record == 90.0


def test_mic_test_parser_rejects_non_positive_duration():
    with pytest.raises(SystemExit):
        build_main_parser().parse_args(["--duration", "0"])


@pytest.mark.parametrize("value", ["nan", "inf", "1e400"])
def test_daemon_parser_rejects_non_finite_max_record(value):
    # `--max-record inf` would reach the daemon's accept-loop `settimeout(inf)`;
    # `nan` would make every recording auto-stop instantly. Stop both at parse time.
    with pytest.raises(SystemExit):
        build_daemon_parser().parse_args(["--max-record", value])


@pytest.mark.parametrize("value", ["nan", "inf", "1e400"])
def test_daemon_parser_rejects_non_finite_idle_timeout(value):
    # `--idle-timeout inf` reached the accept loop's `settimeout()` and killed the
    # daemon. The loop clamps it now, but it should never get that far from a flag.
    with pytest.raises(SystemExit):
        build_daemon_parser().parse_args(["--idle-timeout", value])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_daemon_parser_keeps_the_non_positive_idle_timeout_sentinel(value):
    # `<= 0` means "never unload the model" — the finiteness guard must not eat it.
    args = build_daemon_parser().parse_args(["--idle-timeout", value])
    assert args.idle_timeout == float(value)


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_mic_test_parser_rejects_non_finite_duration(value):
    with pytest.raises(SystemExit):
        build_main_parser().parse_args(["--duration", value])
