"""Shared CLI plumbing for the two entrypoints (`susurro-daemon`, `susurro`).

Both CLIs parse the same engine/audio flags and layer them over the loaded config
the same way; this is the one place that logic lives so adding an engine flag is a
single edit, not shotgun surgery across both `main`s. The daemon adds its own
`--max-record`/`--idle-timeout`/`--no-notify` on top; the mic test adds
`--duration`/`--list-devices`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import Config, pick


def parse_device(value: str) -> int | str:
    """A numeric `--device` is a PortAudio index; anything else is a name substring."""
    return int(value) if value.isdigit() else value


def positive_float(value: str) -> float:
    """An argparse type that rejects non-positive values, so a flag can't smuggle a
    `<= 0` past the config's fail-loud validation (e.g. `--max-record 0` would make
    every recording auto-stop instantly)."""
    parsed = float(value)
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
