# susurro

Local, fully-offline voice dictation for Linux/Wayland/Hyprland, in the spirit of Wispr Flow. Name = "whisper" (es).

**Status: Phase 1 — smoke harness.** Proves the mic → local Whisper (GPU) → text pipeline works and is fast on this machine, behind a clean `Engine`/UI boundary the later phases reuse. Not the real dictation app yet (see `plan/plan.md`).

## What Phase 1 does

A loop that records fixed ~3s audio windows from the mic, transcribes them locally on the GPU with faster-whisper (`large-v3-turbo`, int8, CUDA), applies rule-based cleanup, and prints the result — printing nothing on silence.

> The fixed 3s window is a validation loop, **not** real push-to-talk. It chops words at boundaries by design; that's accepted for a smoke test. Real hold-to-talk comes in Phase 2 (Hyprland global hotkey).

## Requirements

- NVIDIA GPU with CUDA (target: GTX 1060 6GB, Pascal). CPU fallback exists but is slower.
- System `libportaudio` (Arch: `pacman -S portaudio`) for mic capture via `sounddevice`.
- [`uv`](https://docs.astral.sh/uv/) for env + lockfile management.

The CUDA/cuDNN runtime is pinned as pip wheels **inside the venv** (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), isolated from Arch's system CUDA so `pacman -Syu` can't break CTranslate2's ABI.

## Setup

```sh
uv sync
```

## Run

```sh
uv run susurro          # smoke-harness loop; Ctrl-C to quit
```

## Test

```sh
uv run pytest           # formatter unit tests always run;
                        # the engine-transcribe test skips without CUDA + model
```

## Layout

```
src/susurro/
  audio.py        # sounddevice capture: fixed-window 16kHz mono float32
  engine.py       # UI-agnostic Engine: audio -> transcript (warm model)
  formatter.py    # pure rule-based cleanup (unit-tested)
  __main__.py     # smoke-harness loop / entrypoint
scripts/
  gpu_spike.py    # Step 0: prove large-v3-turbo int8 loads + runs on CUDA
tests/
  test_formatter.py
  test_engine.py  # skipped when no CUDA/model
  fixtures/
```
