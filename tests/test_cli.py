"""Unit tests for the shared CLI plumbing (`susurro._cli`).

`apply_engine_audio` is exercised through both entrypoints' `_apply_cli` tests;
here we lock the two small helpers directly (device parsing, the positive-float
guard) and prove the guard actually rejects a non-positive flag at parse time —
the gap where a `--max-record 0` used to slip past the config's fail-loud checks.
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


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_positive_float_rejects_non_positive(value):
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
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
