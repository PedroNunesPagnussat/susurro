# susurro — Spec

## Problem

Voice dictation on Linux/Wayland, fully local: no cloud round-trip, no data leaving the box, works offline. Interaction should feel like a normal key-bind, not a separate app to switch to.

## Approach

A **warm daemon** loads the Whisper model once and keeps it resident in VRAM, because model load dominates latency and this is meant to be used dozens of times a day — paying that cost per utterance would make dictation feel worse than typing. The daemon sits idle until told otherwise, so per-utterance cost is just inference.

Triggering comes from the **compositor** (Hyprland key-bind), not the app in focus and not raw `evdev` — the compositor already sees every key press/release regardless of what window is focused, and doing it this way needs no elevated permissions (`/dev/input` is root-only on this box). A key-bind execs a **thin client** that just writes `start`/`stop` to the daemon's socket; the client carries no model, no audio, nothing slow to import, so the compositor→daemon hop stays cheap even though it's a fresh process per press.

The interaction model is **hold-to-talk**: hold the key while speaking, release to transcribe. This was chosen over a toggle (leaves an easy-to-forget "still recording" state, needs its own on-screen indicator) and over streaming/always-listening (constant compute, wake-word ambiguity, and partial results were never a requirement here). A held key's release is a clean, explicit boundary — the only failure mode is a *missed* release, which is handled below.

On release: capture stops, the audio is transcribed, lightly cleaned up, and typed directly into whatever window is focused — no clipboard, no manual paste. A desktop toast covers the interaction's one usability gap (hold-to-talk has no built-in "am I recording?" cue): it appears the moment recording starts and is replaced by the transcript when it stops.

A **held-key release can be lost** (compositor grabs, chorded modifiers being released out of order) — proven to happen during development. Rather than trying to detect that reliably from the client side (Wayland has no supported "is this key still down" query without `evdev`), the daemon protects itself: a safety timeout force-stops and flushes any recording that runs implausibly long, so a lost release degrades to "that utterance ran long" instead of "the mic is stuck open forever."

## System components

| Component | Role |
|---|---|
| `daemon` | Long-lived process: owns the warm model and the recording state machine, listens on a Unix socket. |
| `ctl` | Thin client (`susurro-ctl start\|stop`) a key-bind execs; one socket write, nothing else. |
| `engine` | Wraps the Whisper model: audio in, formatted text out. |
| `audio` | Mic capture: start/stop recording into a mono float32 buffer. |
| `formatter` | Rule-based text cleanup (whitespace, drop empty/no-speech). |
| `inject` | Types text into the focused window (`wtype`, Wayland's virtual-keyboard protocol). |
| `notify` | Recording/done desktop toasts. |

Everything downstream of "audio in hand" (`engine` → `formatter` → `inject`/`notify`) is orchestrated by `daemon`, not by `ctl` — the client only ever says `start` or `stop`.

## Interfaces & abstractions

The daemon's state machine (`Daemon`) doesn't call the model, the mic, or `wtype` directly — it's constructed with four collaborators, each a narrow structural interface:

- **capturer** (`start() / stop() -> audio`) — real impl: `audio.Recorder`.
- **transcriber** (`transcribe(audio) -> text`) — real impl: `engine.Engine`.
- **inject** (`text -> None`) — a plain callable, not an object; real impl: `inject.inject`.
- **notify** (`recording() / done(text)`) — real impl: `notify.Notifier`, or a no-op `NullNotifier`.

None of these are abstract base classes — they're duck-typed by call signature. The point is substitutability: `Daemon` itself is just "on start, tell the capturer to start; on stop, take its audio, hand it to the transcriber, inject the result, notify either way." Swapping `wtype` for clipboard+paste, or a rule-based formatter for an LLM-based one, means writing a new thing with the same shape — the state machine never changes.

`engine.Engine` exposes exactly one method to the rest of the system, `transcribe(audio: np.ndarray) -> str`. Inside it, a **`Formatter`** (`format(text) -> str`, where `""` means "say nothing") is the seam between "what Whisper produced" and "what gets typed" — today that's just whitespace cleanup, but it's the plug point for smarter post-processing (e.g. an LLM cleanup pass) without touching transcription itself.

The daemon and client don't share code, only a **socket path convention** (`$XDG_RUNTIME_DIR/susurro.sock`) — the client never imports the daemon's (heavy) modules, keeping its startup cheap.

## Configuration

- Trigger key: `Menu` (modifier-free — a modifier chord can drop the release event and orphan a recording).
- Hyprland (`~/.config/hypr/`):
  ```
  bindd = , Menu, Dictate (hold to talk), exec, ~/dev/susurro/.venv/bin/susurro-ctl start
  bindr = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop
  exec-once = uwsm-app -- ~/dev/susurro/.venv/bin/susurro-daemon
  ```
- `susurro-daemon` flags: `--model` (default `large-v3-turbo`, int8/CUDA — the only quantization with good throughput on this GPU), `--device`, `--cpu`, `--max-record` (safety timeout, default 60s), `--no-notify`.
- Injection is `wtype` only; clipboard+paste is a documented manual fallback, not a config flag.
