"""Step 0 — GPU spike. Build-step zero, per the plan.

Proves the GTX 1060 can load + run `large-v3-turbo` int8 on CUDA before we
architect around a warm-in-VRAM model. Prints the transcript, warm per-clip
timing, and observed VRAM. If CUDA fights, fall back to `device="cpu"`.

Run:  uv run python scripts/gpu_spike.py [path/to/audio.wav]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from susurro._cuda import preload_cuda_libs

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "jfk_16k_mono.wav"


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    if not audio.exists():
        print(f"audio not found: {audio}", file=sys.stderr)
        return 2

    loaded = preload_cuda_libs()
    print(f"preloaded {len(loaded)} CUDA libs from venv")

    from faster_whisper import WhisperModel

    mem_before = gpu_mem_used_mib()
    t0 = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    load_s = time.perf_counter() - t0
    mem_after = gpu_mem_used_mib()
    print(f"model loaded in {load_s:.2f}s")
    if mem_before is not None and mem_after is not None:
        print(f"VRAM: {mem_before} -> {mem_after} MiB (delta {mem_after - mem_before} MiB)")

    # Warm-up pass: first inference eats CUDA context + kernel autotune cost.
    list(model.transcribe(str(audio), language="en", vad_filter=True)[0])

    # Timed warm pass — this is the number the Phase-1 DoD cares about.
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio), language="en", vad_filter=True, condition_on_previous_text=False
    )
    text = " ".join(s.text.strip() for s in segments)
    warm_s = time.perf_counter() - t0

    print(f"\ntranscript: {text!r}")
    print(f"warm transcribe: {warm_s:.3f}s  (audio dur {info.duration:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
