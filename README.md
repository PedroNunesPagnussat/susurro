# susurro

Local, fully-offline voice dictation for Linux/Wayland/Hyprland, in the spirit of
Wispr Flow. Name = "whisper" (es).

**Hold a key → speak → release → clean text is typed into the focused window** —
fully local, nothing leaves the box, works offline.

## How it works

A warm **daemon** (`susurro-daemon`) loads the Whisper model once and keeps it
resident, so per-utterance cost is just inference. A Hyprland key-bind execs a thin
**client** (`susurro-ctl`) that writes `start`/`stop` to the daemon's Unix socket;
the compositor sees the key press/release regardless of what window is focused, and
no elevated permissions are needed.

On release the daemon stops capture, transcribes locally on the GPU with
faster-whisper (`large-v3-turbo`, int8, CUDA), applies rule-based cleanup, and
types the result into the focused window via `wtype`. A desktop toast shows the
recording state (hold-to-talk has no built-in "am I recording?" cue). A safety
auto-stop (`--max-record`, default 60s) guarantees a missed release can't wedge the
daemon in "recording".

## Requirements

- NVIDIA GPU with CUDA (target: GTX 1060 6GB, Pascal). CPU fallback exists but is slower.
- System `libportaudio` (Arch: `pacman -S portaudio`) for mic capture via `sounddevice`.
- `wtype` and `notify-send` (libnotify) for injection and toasts.
- [`uv`](https://docs.astral.sh/uv/) for env + lockfile management.

The CUDA/cuDNN runtime is pinned as pip wheels **inside the venv**
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), isolated from Arch's system CUDA so
`pacman -Syu` can't break CTranslate2's ABI.

## Setup

```sh
uv sync
```

## Run

```sh
uv run susurro-daemon        # warm daemon: loads the model, listens on a Unix socket
uv run susurro-ctl start     # begin capture   (bound to key press)
uv run susurro-ctl stop      # end -> transcribe -> type into focused window (key release)
uv run susurro-ctl lang      # flip English <-> Portuguese live (no restart, no reload)
```

`susurro-daemon` flags: `--config` (path to a config file), `--model`, `--device`, `--cpu`,
`--lang`/`--language` (startup language code, e.g. `--lang pt`), `--max-record` (safety
auto-stop seconds), `--idle-timeout` (VRAM-unload idle seconds; `<=0` disables),
`--no-notify`. Every default lives in the config file (see **Configuration**); a flag
only overrides what the file (or the built-in default) set.

## Languages

Language is a per-utterance parameter, so it flips live on the one warm daemon —
no restart, no model reload. `susurro-ctl lang` toggles English↔Portuguese;
`susurro-ctl lang pt` / `lang en` set one explicitly (any faster-whisper code
works). The recording toast shows the active language (`🎙 Recording (pt)…`) and a
transient `🌐 Português` toast confirms each switch. Boot straight into a language
with `susurro-daemon --lang pt`.

There's also a no-daemon mic test that exercises capture + engine directly:

```sh
uv run susurro                 # record 3s windows, transcribe, print; Ctrl-C to quit
uv run susurro --lang pt       # transcribe the windows as Portuguese
uv run susurro --list-devices  # show input devices and exit
```

## Configuration

All tunables (model, compute type, device, language, beam size, VAD, sample rate,
input device, safety timeout, idle-unload, notifications, notify-send backstop) live
in one optional TOML file, **`config.toml` at the repo root** (resolved relative to
the package, so it's found no matter where the daemon is launched from). Precedence is
**built-in defaults < config file < CLI flags** — the file changes the defaults; a
flag still wins over it.

The file is optional and hand-edited: edit the repo's `config.toml` (it's
`.gitignore`d — personal, not pushed) and set only the keys you want to change
(everything else keeps its built-in default). Point the daemon at a different file
with `--config PATH`. A minimal example:

```toml
[engine]
model = "large-v3-turbo"
language = "pt"        # boot into Portuguese
beam_size = 5

[daemon]
max_record_s = 60.0
idle_timeout_s = 300.0  # <= 0 keeps the model resident (no idle-unload)
```

A missing config file is fine (all defaults). A file that exists but is malformed, or
has an invalid value (wrong type / out of range / bad enum), makes susurro **exit with
an error at startup** rather than silently running on a default; unknown keys are
ignored with a warning. There's deliberately no "max window" knob: the capture cap is
derived per caller (daemon `max_record_s` + headroom; the `susurro` mic test
`--duration` + headroom).

### Parameter reference

Delete a line → falls back to the built-in default; a CLI flag still overrides the
file; a bad type / range / enum makes startup fail loud (exit 1); an unknown key only warns.

**`[engine]`**

| Key | Allowed | What it does |
|---|---|---|
| `model` | free string: `tiny`, `base`, `small`, `medium`, `large-v1/v2/v3`, `large-v3-turbo`, `distil-*`, or a path / HF id (add `.en` for English-only) | • Which Whisper model to load<br>• Bigger = more accurate, slower, more VRAM<br>• `large-v3-turbo` = near-large accuracy, much faster |
| `compute_type` | free string; valid set depends on device. **CUDA:** `float16`, `int8_float16`, `int8`, `bfloat16`, `float32`. **CPU:** `int8`, `int16`, `float32` | • Numeric precision of the model<br>• `int8` = smallest/fastest, slight accuracy loss<br>• `float16` = GPU accuracy/speed sweet spot<br>• `float32` = full precision, slowest + most memory<br>• `int8_float16` = mixed (int8 weights, fp16 compute) |
| `device` | enum: `"cuda"` \| `"cpu"` (validated) | • `cuda` = GPU, `cpu` = CPU<br>• CPU works but is much slower |
| `language` | ISO 639-1 2-letter code: `"en"`, `"pt"`, `"es"`, … (~99 supported) | • Forces transcription language (skips auto-detect)<br>• What the live En↔Pt toggle flips |
| `beam_size` | positive int (`>0`) | • Beam search width<br>• Higher = marginally better accuracy, slower<br>• `5` = standard default; `1` = greedy/fastest |
| `vad_filter` | bool: `true` \| `false` | • Strips silence before transcribing (voice-activity detection)<br>• `true` avoids hallucinated text in silent gaps |

**`[audio]`**

| Key | Allowed | What it does |
|---|---|---|
| `sample_rate` | positive int | • Capture rate in Hz<br>• **Keep `16000`** — Whisper is trained on 16 kHz; other values get resampled and hurt quality |
| `channels` | positive int | • `1` = mono (what you want for speech) |
| `device` | int index **or** string name-substring; omit for system default | • Which input mic<br>• Integer = device index; string = matches device name |

**`[daemon]`**

| Key | Allowed | What it does |
|---|---|---|
| `max_record_s` | positive float (`>0`) | • Hard cap on one recording, in seconds<br>• Auto-stops so a forgotten session can't run forever |
| `idle_timeout_s` | any number; `<= 0` disables | • Unload the model from VRAM after this many idle seconds<br>• `<=0` keeps it resident (faster next use, holds VRAM) |
| `notify` | bool: `true` \| `false` | • Whether the daemon sends desktop notifications (recording start/stop, etc.) |

**`[notify]`**

| Key | Allowed | What it does |
|---|---|---|
| `timeout_s` | positive float (`>0`) | • Backstop timeout on the `notify-send` subprocess call (so a hung notifier can't block)<br>• Not how long the popup is shown |

## Hyprland trigger

Pick a **modifier-free, dedicated** key: its release always fires (a modifier chord
can drop the release → stuck recording). A **mouse thumb/side-button** is ideal for
push-to-talk (keyboard stays free); `Menu` works with no extra hardware. Find a
button's code with `wev` (press it, read the `button` number).

In `~/.config/hypr/` (e.g. `bindings.conf`) — `bind` = press, `bindr` = release,
**same key**:

```ini
# modifier-free keyboard key:
bind  = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl start
bindr = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop

# or a mouse thumb button (example: 275 = back; use wev to confirm yours):
# bind  = , mouse:275, exec, ~/dev/susurro/.venv/bin/susurro-ctl start
# bindr = , mouse:275, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop
```

Add a tap-key to flip the language live (Omarchy/Hyprland). This is a normal
`bind` (fires on press), *not* a hold — it just toggles English↔Portuguese; the
hold-to-talk key above is unchanged:

```ini
bind = SUPER SHIFT, D, exec, ~/dev/susurro/.venv/bin/susurro-ctl lang toggle
```

Autostart the warm daemon with the session:

```ini
exec-once = ~/dev/susurro/.venv/bin/susurro-daemon
```

Use the venv's console-script paths (above) rather than `uv run` in Hyprland — no
working-directory or resolution surprises. `hyprctl reload` after editing.

## Injection (and the clipboard fallback)

Text is typed into the focused window with `wtype` (Wayland virtual-keyboard
protocol; Hyprland-native, no uinput / root / `input` group). Empty or
whitespace-only transcripts are a no-op.

If `wtype` misbehaves — very long paragraphs, or an app that drops fast synthetic
keystrokes — the robust fallback is **clipboard + paste**:

```sh
wl-copy "your text"     # then synthesize the app's paste shortcut, e.g. Ctrl+V
```

It's a documented fallback, **not** the default: it clobbers the clipboard and the
paste shortcut varies per app (terminals often use Ctrl+Shift+V).

## Test

```sh
uv run pytest           # pure logic (formatter, buffer, daemon state machine,
                        # client, inject, notify) runs everywhere; the
                        # engine-transcribe test skips without CUDA + model
uv run ruff check src tests
```

## Benchmarking

`susurro-bench` measures which model/runtime is most accurate and fastest **on your
own hardware and voice** (published benchmarks use other speakers, GPUs, and
precisions). It records you reading a fixed set of scripts, runs every available
model (faster-whisper `large-v3-turbo`/`large-v3`, optionally whisper.cpp and NeMo
Parakeet), and prints a comparable WER + latency/RTF table.

```sh
uv sync --extra bench                 # scoring + faster-whisper contenders
uv run susurro-bench --list-models    # what's wired and installed
uv run susurro-bench record           # read the scripts into bench/recordings/ (gitignored)
uv run susurro-bench run              # transcribe every recording with every model -> table
```

Full workflow, metric definitions, and the keep-or-switch results table live in
[`bench/README.md`](bench/README.md).

## Layout

```
src/susurro/
  audio.py        # sounddevice capture (Recorder) + load_wav for offline/eval
  engine.py       # UI-agnostic Engine: audio -> transcript (warm model)
  formatter.py    # pure rule-based cleanup (unit-tested)
  config.py       # single-source config: load config.toml -> Config (CLI overrides)
  _cli.py         # shared CLI plumbing for both entrypoints (flags -> config)
  _cuda.py        # preload the venv's cuBLAS/cuDNN for CTranslate2
  daemon.py       # warm daemon + idle<->recording state machine (Unix socket)
  ctl.py          # thin hold-to-talk client (susurro-ctl start|stop)
  inject.py       # type transcript into focused window via wtype
  notify.py       # recording/done desktop toasts via notify-send
  _ipc.py         # shared socket path (stdlib-only; keeps the client light)
  __main__.py     # no-daemon mic test (susurro)
  bench/          # susurro-bench: offline model benchmark (optional extras)
    transcriber.py  # Transcriber protocol (the common backend seam)
    registry.py     # id -> ModelSpec map (lazy build + import-probe availability)
    faster_whisper_backend.py  whispercpp_backend.py  parakeet_backend.py
    recording.py    # `record`: guided capture + pure save_wav / planner
    wer.py          # WER/CER scoring via jiwer (normalized + raw + CER)
    runner.py       # `run`: measure (warm-up discarded, median of N) + score + report
    cli.py          # susurro-bench entrypoint (record / run / --list-models)
config.toml       # repo-local tunables (optional, .gitignore'd; --config to relocate)
bench/scripts/    # committed reference scripts; recordings/ + results/ gitignored
tests/
  test_formatter.py  test_audio.py    test_daemon.py  test_ctl.py
  test_inject.py     test_notify.py   test_config.py  test_main.py
  test_cli.py        test_cuda.py     test_serve.py   test_lazy_engine.py
  test_engine.py     # skipped when no CUDA/model
  test_bench_registry.py  test_bench_transcriber.py  test_bench_cli.py
  test_bench_recording.py  test_bench_wer.py  test_bench_runner.py
  test_bench_faster_whisper.py  # skipped when no CUDA/model
  fixtures/
```