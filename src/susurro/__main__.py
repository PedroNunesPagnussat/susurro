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

import numpy as np

from ._cli import (
    add_common_flags,
    apply_engine_audio,
    log,
    positive_seconds,
    preflight_language,
    start_engine,
)
from .audio import Recorder, list_input_devices
from .config import ConfigError, load_config


def _build_parser() -> argparse.ArgumentParser:
    # `--duration` is a mic-test-only knob; the rest are the shared engine/audio
    # flags (defaults < config file < CLI flag).
    p = argparse.ArgumentParser(prog="susurro", description=__doc__)
    add_common_flags(p)
    p.add_argument(
        "--duration", type=positive_seconds, default=3.0, help="window length in seconds"
    )
    p.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    return p


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
        # No wrapper: the mic test adds no flags of its own beyond the shared
        # engine/audio ones, so it calls the shared resolver directly.
        config = apply_engine_audio(load_config(args.config), args)
    except ConfigError as exc:
        log(str(exc))
        return 1

    eng, aud = config.engine, config.audio
    if not preflight_language(eng.language):
        return 1

    print(f"loading {eng.model} on {eng.device} ({eng.language}) ...", flush=True)
    engine = start_engine(eng, sample_rate=aud.sample_rate)
    if engine is None:
        return 1

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
