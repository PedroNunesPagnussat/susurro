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

from .audio import SAMPLE_RATE, Recorder, list_input_devices
from .engine import DEFAULT_MODEL, Engine


def _parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="susurro", description=__doc__)
    p.add_argument("--duration", type=float, default=3.0, help="window length in seconds")
    p.add_argument("--device", type=_parse_device, default=None, help="input device index or name")
    p.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model name")
    p.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
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

    device = "cpu" if args.cpu else "cuda"
    print(f"loading {args.model} on {device} ...", flush=True)
    engine = Engine(args.model, device=device)

    # Warm the CUDA kernels so the first real window already hits warm timing.
    engine.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    recorder = Recorder(max_duration_s=args.duration + 1.0, device=args.device)
    print(f"ready — input: {args.device or 'default'}. Speak; Ctrl-C to quit.", flush=True)

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
