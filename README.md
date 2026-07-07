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

See `SPEC.md` for the full design rationale.

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

`susurro-daemon` flags: `--model` (default `large-v3-turbo`), `--device`, `--cpu`,
`--lang`/`--language` (startup language code, default `en`; e.g. `--lang pt`),
`--max-record` (safety auto-stop seconds, default 60), `--no-notify`.

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

## Layout

```
src/susurro/
  audio.py        # sounddevice capture (Recorder) + load_wav for offline/eval
  engine.py       # UI-agnostic Engine: audio -> transcript (warm model)
  formatter.py    # pure rule-based cleanup (unit-tested)
  _cuda.py        # preload the venv's cuBLAS/cuDNN for CTranslate2
  daemon.py       # warm daemon + idle<->recording state machine (Unix socket)
  ctl.py          # thin hold-to-talk client (susurro-ctl start|stop)
  inject.py       # type transcript into focused window via wtype
  notify.py       # recording/done desktop toasts via notify-send
  _ipc.py         # shared socket path (stdlib-only; keeps the client light)
  __main__.py     # no-daemon mic test (susurro)
tests/
  test_formatter.py  test_audio.py  test_daemon.py  test_ctl.py
  test_inject.py     test_notify.py
  test_engine.py     # skipped when no CUDA/model
  fixtures/
```