"""`susurro` — a no-daemon mic test: record a fixed window, transcribe, print.

Exercises capture + engine without the daemon/socket/hotkey, using the same
`Recorder` the daemon uses. Handy for checking the mic and model in isolation;
the real dictation app is `susurro-daemon` + `susurro-ctl`.

    uv run susurro                 # record 3s, transcribe, print (loops until Ctrl-C)
    uv run susurro --list-devices  # show input devices and exit
    uv run susurro --device 4      # pick an input by index or name substring
    uv run susurro --cpu           # CPU fallback
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from .audio import Recorder, list_input_devices
from .config import Config, ConfigError, load_config, pick
from .engine import Engine


def _parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _build_parser() -> argparse.ArgumentParser:
    # Flag defaults are None so `_apply_cli` only overrides config when passed
    # (defaults < config file < CLI flag). `--duration` is a mic-test-only knob.
    p = argparse.ArgumentParser(prog="susurro", description=__doc__)
    p.add_argument("--config", default=None, help="path to config.toml")
    p.add_argument("--duration", type=float, default=3.0, help="window length in seconds")
    p.add_argument("--device", type=_parse_device, default=None, help="input device index or name")
    p.add_argument("--model", default=None, help="faster-whisper model name")
    p.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    p.add_argument(
        "--lang", "--language", dest="lang", default=None, help="transcription language code (e.g. en, pt)"
    )
    p.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    return p


def _apply_cli(config: Config, args: argparse.Namespace) -> Config:
    """Layer the mic test's flags over the loaded config (`--cpu` forces CPU)."""
    engine = replace(
        config.engine,
        model=pick(args.model, config.engine.model),
        device="cpu" if args.cpu else config.engine.device,
        language=pick(args.lang, config.engine.language),
    )
    audio = replace(config.audio, device=pick(args.device, config.audio.device))
    return replace(config, engine=engine, audio=audio)


def _record_window(recorder: Recorder, duration_s: float) -> np.ndarray:
    recorder.start()
    time.sleep(duration_s)
    return recorder.stop()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_devices:
        for idx, name in list_input_devices():
            print(f"  [{idx}] {name}")
        return 0

    try:
        config = _apply_cli(load_config(args.config), args)
    except ConfigError as exc:
        print(f"susurro: {exc}", file=sys.stderr, flush=True)
        return 1

    eng, aud = config.engine, config.audio
    print(f"loading {eng.model} on {eng.device} ({eng.language}) ...", flush=True)
    engine = Engine(
        eng.model,
        device=eng.device,
        compute_type=eng.compute_type,
        language=eng.language,
        beam_size=eng.beam_size,
        vad_filter=eng.vad_filter,
    )

    # Warm the CUDA kernels so the first real window already hits warm timing.
    engine.transcribe(np.zeros(aud.sample_rate // 2, dtype=np.float32))

    recorder = Recorder(
        max_duration_s=args.duration + 1.0,
        samplerate=aud.sample_rate,
        channels=aud.channels,
        device=aud.device,
    )
    print(f"ready — input: {aud.device or 'default'}. Speak; Ctrl-C to quit.", flush=True)

    in_silence = False
    try:
        while True:
            audio = _record_window(recorder, args.duration)
            t0 = time.perf_counter()
            text = engine.transcribe(audio)
            dt = time.perf_counter() - t0

            if text:
                if in_silence:
                    print()  # close the quiet-marker streak
                    in_silence = False
                print(f"{text}   ({dt:.2f}s)", flush=True)
            else:
                print("·", end="", flush=True)  # quiet marker on silence
                in_silence = True
    except KeyboardInterrupt:
        print("\nbye")
        return 0


if __name__ == "__main__":
    sys.exit(main())
