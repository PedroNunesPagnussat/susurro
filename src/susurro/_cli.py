"""Shared CLI plumbing for the two entrypoints (`susurro-daemon`, `susurro`).

Both CLIs parse the same engine/audio flags and layer them over the loaded config
the same way; this is the one place that logic lives, so adding a flag is a single
edit, not shotgun surgery across both `main`s. The daemon adds its own
`--max-record`/`--idle-timeout`/`--no-notify` on top; the mic test adds
`--duration`/`--list-devices`.

Deliberately *not* here: how a config becomes an engine. That lives in `engine.py`
(`build_engine`/`load_engine`) so an engine change doesn't touch the CLI module;
`start_engine` below is only the presentation half — turning the failure into the
same `susurro: …` line every other startup error uses.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace

from .config import MAX_SECONDS, Config, EngineConfig, pick
from .engine import Engine, EngineLoadError, LazyEngine, is_supported_language, load_engine


def log(msg: str) -> None:
    """Write one `susurro: …` line to stderr — where both entrypoints shape their
    output, so a config error, a rejected language, a failed model load and the
    daemon's loop warnings all read the same. (`config._warn` is the one forced
    duplicate: `_cli` imports `config`, so `config` can't import back.) Flushed: the
    daemon is started from Hyprland's `exec-once`, where stderr is a pipe, and a
    buffered tail would be lost exactly when something went wrong."""
    print(f"susurro: {msg}", file=sys.stderr, flush=True)


def parse_device(value: str) -> int | str:
    """A numeric `--device` is a PortAudio index; anything else is a name substring."""
    return int(value) if value.isdigit() else value


def seconds(value: str) -> float:
    """An argparse duration: any finite value up to `MAX_SECONDS`, sign included.

    For flags where a non-positive value is meaningful (`--idle-timeout 0` disables
    idle-unload, so `positive_seconds` would be wrong) but an unusable magnitude
    never is. Both ends of the range matter: `nan`/`inf` survive every `<= 0` check,
    and a huge-but-finite value like `1e10` survives a finiteness check — either one
    then reaches `settimeout()`/`sleep()`, which raise `OverflowError` and take the
    process down (see `config.MAX_SECONDS`)."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be a finite number (got {value})")
    if parsed > MAX_SECONDS:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SECONDS:.0f} seconds (got {value})")
    return parsed


def positive_seconds(value: str) -> float:
    """An argparse duration that must also be > 0, so a flag can't smuggle past the
    config's fail-loud validation (e.g. `--max-record 0` would make every recording
    auto-stop instantly). `float("1e400")` is already `inf`, so the range checks in
    `seconds` do the rest."""
    parsed = seconds(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {value})")
    return parsed


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Add the engine/audio flags shared by both CLIs. Defaults are None ('not
    passed') so `apply_engine_audio` only overrides config when a flag is given:
    built-in defaults < config file < CLI flag."""
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--model", default=None, help="faster-whisper model name")
    parser.add_argument(
        "--device", type=parse_device, default=None, help="input device index or name"
    )
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    parser.add_argument(
        "--lang",
        "--language",
        dest="lang",
        default=None,
        help="transcription language code (e.g. en, pt)",
    )


def apply_engine_audio(config: Config, args: argparse.Namespace) -> Config:
    """Layer the shared engine/audio flags over the loaded config (defaults < file <
    flag). `--cpu` is a force-off switch (no matching on-flag), so it overrides the
    config only toward CPU."""
    engine = replace(
        config.engine,
        model=pick(args.model, config.engine.model),
        device="cpu" if args.cpu else config.engine.device,
        language=pick(args.lang, config.engine.language),
    )
    audio = replace(config.audio, device=pick(args.device, config.audio.device))
    return replace(config, engine=engine, audio=audio)


def preflight_language(code: str) -> bool:
    """True if Whisper will accept `code`; else log why and return False.

    Whisper validates the language only at the warmup transcribe, where a typo'd
    `--lang` surfaces as a failed *model* load, blaming the model name and inlining
    all 100 accepted codes. Both entrypoints check here first. Fails open (see
    `engine.is_supported_language`)."""
    if is_supported_language(code):
        return True
    log(f"unsupported language {code!r} — check --lang / [engine] language")
    return False


def start_engine(
    cfg: EngineConfig, *, sample_rate: int, lazy: bool = False
) -> Engine | LazyEngine | None:
    """`engine.load_engine`, with the failure reported as a `susurro: …` line + None.

    The split is deliberate: building/warming a model is `engine.py`'s job, shaping
    a startup failure for a terminal is this module's. Reported like a config error
    (same line, same exit 1) instead of dumping a stack trace into a log nobody reads
    — under `exec-once` that traceback goes nowhere."""
    try:
        return load_engine(cfg, sample_rate=sample_rate, lazy=lazy)
    except EngineLoadError as exc:
        log(str(exc))
        return None
