# susurro

Local, fully-offline voice dictation for Linux/Wayland/Hyprland, in the spirit of Wispr Flow. Name = "whisper" (es).

**Status: Phase 1 done; Phase 2 (hold-to-talk) in progress.** Phase 1 proved the mic → local Whisper (GPU) → text pipeline works and is fast on this machine, behind a clean `Engine`/UI boundary the later phases reuse. Phase 2 builds the real system-wide dictation app on top (see [Phase 2 below](#phase-2--hold-to-talk-in-progress) and `plan/plan.md`).

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

## Phase 2 — hold-to-talk (in progress)

Real system-wide dictation: **hold a key → speak → release → clean text is typed
into the focused window**, fully local. Split into a warm **daemon** (owns the
model) and a thin **client** the compositor triggers on key press/release.

```sh
uv run susurro-daemon        # warm daemon: loads the model, listens on a Unix socket
uv run susurro-ctl start     # begin capture   (bound to key press)
uv run susurro-ctl stop      # end -> transcribe -> type into focused window (key release)
```

The daemon loads + warms the model once, so per-utterance latency is
inference-bound. A safety auto-stop (`--max-record`, default 30s) guarantees a
missed release can't wedge it in "recording". `susurro-ctl` is stdlib-only and
tiny so the trigger stays snappy.

### Hyprland trigger

Pick a **modifier-free, dedicated** key: its release always fires (a modifier
chord can drop the release → stuck recording). A **mouse thumb/side-button** is
ideal for push-to-talk (keyboard stays free); `Menu` works with no extra
hardware. Find a button's code with `wev` (press it, read the `button` number).

In `~/.config/hypr/` (e.g. `bindings.conf`) — `bind` = press, `bindr` = release,
**same key**:

```ini
# mouse thumb button (example: 275 = back; use wev to confirm yours)
bind  = , mouse:275, exec, ~/dev/susurro/.venv/bin/susurro-ctl start
bindr = , mouse:275, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop

# or a modifier-free keyboard key:
# bind  = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl start
# bindr = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop
```

Autostart the warm daemon with the session:

```ini
exec-once = ~/dev/susurro/.venv/bin/susurro-daemon
```

Use the venv's console-script paths (above) rather than `uv run` in Hyprland —
no working-directory or resolution surprises. `hyprctl reload` after editing.

### Injection (and the clipboard fallback)

Text is typed into the focused window with `wtype` (Wayland virtual-keyboard
protocol; Hyprland-native, no uinput / root / `input` group). Empty or
whitespace-only transcripts are a no-op.

If `wtype` misbehaves for you — very long paragraphs, or an app that drops fast
synthetic keystrokes — the robust fallback is **clipboard + paste**:

```sh
wl-copy "your text"     # then synthesize the app's paste shortcut, e.g. Ctrl+V
```

It's a documented fallback, **not** the default: it clobbers the clipboard and
the paste shortcut varies per app (terminals often use Ctrl+Shift+V). `wtype`
stays the default.

## Test

```sh
uv run pytest           # pure logic (formatter, buffer, daemon state machine,
                        # client) runs everywhere; the engine-transcribe test
                        # skips without CUDA + model
uv run ruff check src tests
```

## Layout

```
src/susurro/
  audio.py        # sounddevice capture: fixed-window + variable-length (Recorder)
  engine.py       # UI-agnostic Engine: audio -> transcript (warm model)
  formatter.py    # pure rule-based cleanup (unit-tested)
  daemon.py       # Phase 2: warm daemon + idle<->recording state machine (Unix socket)
  ctl.py          # Phase 2: thin hold-to-talk client (susurro-ctl start|stop)
  inject.py       # Phase 2: type transcript into focused window via wtype
  _ipc.py         # shared socket path (stdlib-only; keeps the client light)
  __main__.py     # Phase 1 smoke-harness loop / entrypoint
scripts/
  gpu_spike.py     # Step 0: prove large-v3-turbo int8 loads + runs on CUDA
  trigger_spike.py # Step 6: Hyprland bind/bindr -> socket -> wtype spike
tests/
  test_formatter.py  test_audio.py  test_daemon.py  test_ctl.py  test_inject.py
  test_engine.py     # skipped when no CUDA/model
  fixtures/
```
