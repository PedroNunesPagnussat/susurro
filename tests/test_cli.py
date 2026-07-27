"""Unit tests for the shared CLI plumbing (`susurro._cli`).

`apply_engine_audio` is exercised through the daemon's `_apply_cli` tests (see
tests/test_daemon.py); here we lock the small helpers directly (device parsing, the
duration guards, the startup engine load) and prove the guards actually reject a
non-positive, non-finite or absurdly large flag at parse time — the gap where a
`--max-record 0` (or `inf`, or `1e10`) used to slip past the config's fail-loud checks.
"""

import argparse

import numpy as np
import pytest

from susurro import _cli
from susurro.__main__ import _build_parser as build_main_parser
from susurro.__main__ import main as mic_test_main
from susurro.config import MAX_SECONDS, EngineConfig
from susurro.daemon import _build_parser as build_daemon_parser
from susurro.daemon import main as daemon_main


@pytest.mark.parametrize(
    ("value", "expected"), [("4", 4), ("0", 0), ("hw:1", "hw:1"), ("USB mic", "USB mic")]
)
def test_parse_device_int_vs_name(value, expected):
    assert _cli.parse_device(value) == expected


def test_positive_seconds_accepts_positive():
    assert _cli.positive_seconds("2.5") == 2.5


@pytest.mark.parametrize("value", ["0", "-1", "2.5"])
def test_seconds_accepts_any_in_range_value(value):
    # Unlike `positive_seconds`, non-positive is legal here: `--idle-timeout 0` is the
    # documented way to keep the model resident.
    assert _cli.seconds(value) == float(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_seconds_rejects_non_finite(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _cli.seconds(value)


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_positive_seconds_rejects_non_positive(value):
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        _cli.positive_seconds(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity", "1e400"])
def test_positive_seconds_rejects_non_finite(value):
    # `float()` happily returns nan/inf (and silently overflows `1e400` to inf), and
    # every `<= 0` comparison against them is False — so they need their own check.
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _cli.positive_seconds(value)


@pytest.mark.parametrize("guard", [_cli.seconds, _cli.positive_seconds])
def test_duration_guards_reject_huge_but_finite_values(guard):
    # Finiteness alone isn't enough: `settimeout()`/`sleep()` raise OverflowError
    # above ~9.2e9 seconds, so `1e10` is finite, passes every range check, and then
    # kills the process from inside the accept loop / the mic-test sleep.
    with pytest.raises(argparse.ArgumentTypeError, match="at most"):
        guard("1e10")


@pytest.mark.parametrize("guard", [_cli.seconds, _cli.positive_seconds])
def test_duration_guards_accept_the_cap_itself(guard):
    assert guard(str(MAX_SECONDS)) == MAX_SECONDS


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


@pytest.mark.parametrize("value", ["nan", "inf", "1e400", "1e10"])
def test_daemon_parser_rejects_unusable_idle_timeout(value):
    # `--idle-timeout inf` (and `1e10`) reached the accept loop's `settimeout()` and
    # killed the daemon. `_serve_once` caps its timeout now as a backstop, but a flag
    # this wrong should never get that far.
    with pytest.raises(SystemExit):
        build_daemon_parser().parse_args(["--idle-timeout", value])


@pytest.mark.parametrize("flag", ["--max-record", "--idle-timeout"])
def test_daemon_parser_rejects_huge_but_finite_durations(flag):
    with pytest.raises(SystemExit):
        build_daemon_parser().parse_args([flag, "1e10"])


def test_mic_test_parser_rejects_a_huge_duration():
    # `--duration 1e10` reaches `time.sleep`, which raises OverflowError the same way.
    with pytest.raises(SystemExit):
        build_main_parser().parse_args(["--duration", "1e10"])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_daemon_parser_keeps_the_non_positive_idle_timeout_sentinel(value):
    # `<= 0` means "never unload the model" — the finiteness guard must not eat it.
    args = build_daemon_parser().parse_args(["--idle-timeout", value])
    assert args.idle_timeout == float(value)


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_mic_test_parser_rejects_non_finite_duration(value):
    with pytest.raises(SystemExit):
        build_main_parser().parse_args(["--duration", value])


# --- startup language pre-flight -------------------------------------------
#
# Whisper validates the language inside its tokenizer, i.e. at the warmup transcribe.
# Left that late, both entrypoints report a typo'd `--lang` through their model-load
# handler — "failed to load model 'large-v3-turbo' on cuda: 'xx' is not a valid
# language code (accepted language codes: af, am, ar, ...)" — which blames the model
# and inlines all 100 codes. Both must reject it before any Engine is built, which is
# also what makes these tests runnable without CUDA or a model.


def _config(tmp_path, body: str = "") -> str:
    """A throwaway config file, so these never read the repo's own config.toml."""
    path = tmp_path / "config.toml"
    path.write_text(body)
    return str(path)


@pytest.mark.parametrize(
    "entrypoint", [daemon_main, mic_test_main], ids=["susurro-daemon", "susurro"]
)
def test_startup_rejects_an_unsupported_lang_flag(entrypoint, tmp_path, capsys):
    assert entrypoint(["--config", _config(tmp_path), "--lang", "xx"]) == 1
    err = capsys.readouterr().err
    assert "xx" in err  # names the offending code, not the model
    assert "unsupported language" in err
    assert "is not a valid language code" not in err  # not the model's late error


@pytest.mark.parametrize(
    "entrypoint", [daemon_main, mic_test_main], ids=["susurro-daemon", "susurro"]
)
def test_startup_rejects_an_unsupported_config_language(entrypoint, tmp_path, capsys):
    # Same gap via the other input: `[engine] language` is validated as a string only.
    config = _config(tmp_path, '[engine]\nlanguage = "english"\n')
    assert entrypoint(["--config", config]) == 1
    assert "english" in capsys.readouterr().err


# --- startup engine load ----------------------------------------------------
#
# `start_engine` is the shared contract both entrypoints rely on: a startup failure
# becomes one `susurro: …` line + exit 1, never a traceback into a log nobody reads
# (under `exec-once`, stderr goes nowhere at all). `engine.load_engine` does the
# building and warming; these patch it so the path is testable without CUDA.

_CFG = EngineConfig(model="tiny", device="cuda")


class _FakeEngine:
    def __init__(self):
        self.warmed = None
        self.language = None

    def transcribe(self, audio):
        self.warmed = audio
        return ""

    def set_language(self, code):  # LazyEngine re-applies it on every (re)build
        self.language = code


def test_start_engine_returns_a_warmed_engine(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(_cli, "load_engine", lambda cfg, **kw: fake)
    assert _cli.start_engine(_CFG, sample_rate=16_000) is fake


def test_start_engine_reports_a_failed_load_and_returns_none(monkeypatch, capsys):
    def boom(cfg, **kw):
        raise _cli.EngineLoadError("failed to load model 'tiny' on cuda: no CUDA driver")

    monkeypatch.setattr(_cli, "load_engine", boom)
    assert _cli.start_engine(_CFG, sample_rate=16_000) is None

    err = capsys.readouterr().err
    assert err.startswith("susurro: ")  # same shape as every other startup error
    assert "failed to load model 'tiny' on cuda" in err  # names the two knobs
    assert "Traceback" not in err


def test_build_engine_maps_every_config_field_onto_the_constructor(monkeypatch):
    seen = {}

    def fake_engine(model, **kwargs):
        seen.update(model=model, **kwargs)
        return _FakeEngine()

    monkeypatch.setattr("susurro.engine.Engine", fake_engine)
    from susurro.engine import build_engine

    cfg = EngineConfig(
        model="tiny",
        compute_type="float16",
        device="cpu",
        language="pt",
        beam_size=1,
        vad_filter=False,
    )
    build_engine(cfg)
    assert seen == {
        "model": "tiny",
        "compute_type": "float16",
        "device": "cpu",
        "language": "pt",
        "beam_size": 1,
        "vad_filter": False,
    }


def test_build_engine_lazy_defers_the_build_and_stays_unloadable(monkeypatch):
    # The daemon's idle-unload only exists if it's handed a LazyEngine: `Daemon`
    # disables the feature outright for an engine that can't unload, silently. So
    # pin the shape, not just that something engine-ish came back.
    from susurro.engine import LazyEngine, build_engine

    built = []
    monkeypatch.setattr(
        "susurro.engine.Engine", lambda model, **kw: built.append(model) or _FakeEngine()
    )

    engine = build_engine(EngineConfig(model="tiny"), lazy=True)

    assert isinstance(engine, LazyEngine)
    assert built == []  # nothing constructed until the first load/transcribe
    engine.load()
    assert built == ["tiny"] and engine.loaded
    engine.unload()
    assert not engine.loaded


def test_load_engine_warms_the_model_before_returning(monkeypatch):
    # The warmup is the point: without it a LazyEngine defers the build, and a bad
    # model name surfaces one utterance later instead of at startup.
    from susurro.engine import load_engine

    fake = _FakeEngine()
    monkeypatch.setattr("susurro.engine.build_engine", lambda cfg, **kw: fake)

    assert load_engine(_CFG, sample_rate=16_000) is fake
    assert isinstance(fake.warmed, np.ndarray)
    assert len(fake.warmed) == 8_000  # half a second of silence
    assert fake.warmed.dtype == np.float32


@pytest.mark.parametrize("failing", ["build", "warmup"])
def test_load_engine_wraps_either_failure_with_the_model_and_device(monkeypatch, failing):
    # Both halves fail the same way to the caller: a bad model name blows up in the
    # constructor, a broken CUDA install in the warmup transcribe.
    from susurro.engine import EngineLoadError, load_engine

    class _Warmup(_FakeEngine):
        def transcribe(self, audio):
            raise RuntimeError("no CUDA driver")

    def build(cfg, **kw):
        if failing == "build":
            raise RuntimeError("no CUDA driver")
        return _Warmup()

    monkeypatch.setattr("susurro.engine.build_engine", build)
    with pytest.raises(EngineLoadError, match=r"failed to load model 'tiny' on cuda") as exc:
        load_engine(_CFG, sample_rate=16_000)
    assert "no CUDA driver" in str(exc.value)  # keeps the underlying cause
    assert exc.value.__cause__ is not None  # chained, so a debugger still gets it
