"""Phase-1 smoke-harness loop: capture -> transcribe -> format -> print.

Records fixed ~3s windows (NOT push-to-talk — that's Phase 2 via Hyprland), prints
each transcript with its warm timing, and a quiet marker on silence. Ctrl-C quits.

    uv run susurro                 # default: 3s windows, GPU
    uv run susurro --list-devices  # show input devices and exit
    uv run susurro --device 4      # pick an input by index or name substring
    uv run susurro --cpu           # CPU fallback
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .audio import SAMPLE_RATE, default_input_device, list_input_devices, record_window
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

    src = default_input_device() if args.device is None else args.device
    print(f"ready — input: {src}. Speak; Ctrl-C to quit.", flush=True)

    in_silence = False
    try:
        while True:
            audio = record_window(args.duration, device=args.device)
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
