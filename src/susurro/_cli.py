"""Shared CLI plumbing for the two entrypoints (`susurro-daemon`, `susurro`).

Both CLIs parse the same engine/audio flags, layer them over the loaded config the
same way, and start the model the same way; this is the one place that logic lives
so adding an engine knob is a single edit, not shotgun surgery across both `main`s.
The daemon adds its own `--max-record`/`--idle-timeout`/`--no-notify` on top; the
mic test adds `--duration`/`--list-devices`.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .config import Config, EngineConfig, pick
from .engine import Engine, is_supported_language


def log(msg: str) -> None:
    """Write one `susurro: …` line to stderr — the single place both entrypoints
    shape their output, so a config error, a rejected language, a failed model load
    and the daemon's loop warnings all read the same. Flushed: the daemon is started
    from Hyprland's `exec-once`, where stderr is a pipe, and a buffered tail would be
    lost exactly when something went wrong."""
    print(f"susurro: {msg}", file=sys.stderr, flush=True)


def parse_device(value: str) -> int | str:
    """A numeric `--device` is a PortAudio index; anything else is a name substring."""
    return int(value) if value.isdigit() else value


def finite_float(value: str) -> float:
    """An argparse type that rejects `nan`/`inf` but accepts any finite value, sign
    included. For flags where a non-positive value is meaningful (`--idle-timeout 0`
    disables idle-unload, so `positive_float` would be wrong) but a non-finite one
    never is: it survives every range check, then reaches the daemon's accept-loop
    `settimeout()`, which raises `OverflowError` on a non-finite timeout."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be a finite number (got {value})")
    return parsed


def positive_float(value: str) -> float:
    """An argparse type that rejects non-positive and non-finite values, so a flag
    can't smuggle past the config's fail-loud validation (e.g. `--max-record 0` would
    make every recording auto-stop instantly). `nan`/`inf` need their own check: every
    `<= 0` comparison against them is False, and `float("1e400")` is already `inf`."""
    parsed = finite_float(value)
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


def build_engine(cfg: EngineConfig) -> Engine:
    """`Engine` from the resolved `[engine]` config — the one place those fields map
    onto the constructor, so a new knob is a single edit."""
    return Engine(
        cfg.model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
    )


def load_engine[E](cfg: EngineConfig, build: Callable[[], E], *, sample_rate: int) -> E | None:
    """Build the engine and warm it, or report the failure and return None.

    `build` absorbs the entrypoints' one difference (the mic test wants an `Engine`,
    the daemon a `LazyEngine` around one). The warmup is where a bad model name,
    failed download or broken CUDA install surfaces — a `LazyEngine` defers the load
    until the first transcribe — and it leaves the kernels compiled, so the first
    real utterance already hits warm timing. `E` needs only a `transcribe`."""
    try:
        engine = build()
        engine.transcribe(np.zeros(sample_rate // 2, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001 (a startup failure gets a message, not a traceback)
        # Report it like a config error (same `susurro: …` + exit 1) instead of
        # dumping a stack trace into a log nobody reads. Name the model and the
        # device — they're the knobs.
        log(f"failed to load model {cfg.model!r} on {cfg.device}: {exc}")
        return None
    return engine
