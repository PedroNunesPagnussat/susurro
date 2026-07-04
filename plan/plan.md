# susurro — Plan

Local, fully-offline voice dictation for Linux/Wayland/Hyprland, in the spirit of Wispr Flow. Name = "whisper" (es).

## Background

Greenfield project at `/home/pedro/dev/susurro` (sibling to other projects under `~/dev`; not yet a git repo). Target machine, verified via probes:

- **OS/session:** Arch Linux, Wayland, Hyprland compositor. Shell zsh. Currently operating **inside tmux** (`TERM=tmux-256color`) — this rules out reliable terminal key-release detection (kitty keyboard protocol is mangled by tmux).
- **GPU:** NVIDIA GTX 1060 6GB (Pascal). Weak FP16 throughput but has DP4A, so **int8 is the right quantization**. `nvidia-smi` present.
- **Audio:** PipeWire (`pw-record`, `pactl` present). Capture will go through PortAudio → ALSA → PipeWire (`pipewire-alsa` compat).
- **Toolchain:** Python 3.13.7, Rust/cargo 1.95 available. `uv` will manage the Python env.
- **Permissions:** user is **not** in the `input` group; `/dev/input/event*` is `root:input`. Only matters for Phase 2 (global hotkey via evdev).

Design was settled through an extensive grill session (see Decisions). This plan implements **Phase 1 only** (the smoke harness); Phases 2–3 are scoped at a high level for continuity.

## Problem Statement

I want to dictate by voice on my Linux/Wayland machine and get clean text, fully locally (no cloud, no data leaving the box). Before building the real system-wide dictation app, I need to prove the core pipeline — mic → local Whisper on GPU → text — actually works on this specific old GPU, and feels fast enough to build a warm-model architecture around.

## Solution

A phased build:

- **Phase 1 (this plan): smoke harness.** A Python loop that records fixed ~3s audio windows from the mic, transcribes them locally on the GPU with faster-whisper, applies tiny rule-based cleanup, and prints the result — printing nothing on silence. Its only job is to *prove the pipeline works and is fast*, and to establish a clean `Engine`/UI boundary the later phases reuse.
- **Phase 2 (future): the real app.** Global-hotkey push-to-talk (hold=record, release=transcribe) driven by Hyprland, a daemon/client split over a Unix socket, injection into the focused Wayland window, and a pluggable local-LLM cleanup stage.
- **Phase 3 (future, optional): cloud** transcription/cleanup backend behind the same formatter/engine seams.

## Decisions

- **Interaction model = push-to-talk (hold), batch.** Record while trigger held, transcribe on release. No streaming, ever (explicitly dropped as a requirement). Explicit trigger avoids false activations and keeps the state machine trivial. Ruled out: toggle (leave-it-recording risk + needs an always-on-top Wayland indicator), VAD/always-listening (constant compute, wake-word problem).
- **Phase 1 trigger is NOT real push-to-talk.** Terminals (esp. under tmux) don't report key-release, so the harness uses **fixed ~3s capture windows** purely as a validation loop. Real hold-to-talk is deferred to Phase 2 where the trigger comes from the **compositor** (Hyprland `bind`), not the terminal. Accepted caveat: fixed windows chop words at boundaries — fine for a smoke test, not to be judged for accuracy.
- **Transcription engine = faster-whisper (CTranslate2).** Fastest practical accuracy/watt, built-in Silero VAD, mature. Ruled out: whisper.cpp (Pascal FP16 path slower, CUDA setup fussier — reconsider only if we go Rust), cloud (violates the local requirement).
- **Model = `large-v3-turbo`, `compute_type="int8"`, `device="cuda"`.** ~1.5–2GB VRAM, near-large accuracy, fast decode; int8 leans on Pascal's DP4A. `distil-large-v3.5` (English-only, faster) is a drop-in swap later if more speed is wanted — no lock-in.
- **English-only.** `language="en"` hard-set to skip detection time and misdetection.
- **Transcription defaults (consciously accepted):** `vad_filter=True` (drops non-speech → prevents Whisper's silence-hallucination, critical for a loop that captures lots of silence), `condition_on_previous_text=False` (independent utterances → no repetition/context bleed), `beam_size=5` (fine on turbo for short clips; drop to 1 only if latency bites).
- **Cleanup = pluggable formatter stage, rule-based in Phase 1.** Interface exists from day 1; Phase 1's only impl trims/collapses whitespace and drops empty/`no_speech` results. **No filler-word removal** (regex eats real words; "the sum" ≠ filler). Local LLM cleanup (~3B Q4, fits alongside turbo in 6GB) is a Phase 2 formatter impl. Ruled out: LLM in Phase 1 (VRAM contention + edit-fidelity risk before the core loop is proven).
- **Language/stack = Python, single long-lived process.** faster-whisper is Python anyway. Code split into a UI-agnostic `Engine` core (capture → VAD → transcribe → format → emit text) + thin UI/driver. Ruled out for now: daemon+client+IPC (YAGNI until Phase 2's external trigger forces it — the `Engine` extracts behind a Unix socket mechanically then), Rust/hybrid (hold-to-talk latency is dominated by inference, not client language).
- **Audio = `sounddevice` (PortAudio), in-process, 16kHz mono float32, callback-based.** Numpy arrays feed faster-whisper directly with no resampling; the callback/start-stop model is reused verbatim in Phase 2. Ruled out: `pw-record` subprocess (clumsy start/stop lifecycle, would be rewritten for P2), `soundcard` (less battle-tested).
- **Env = `uv`; CUDA/cuDNN via pinned `nvidia-*-cu12` pip wheels inside the venv.** The venv owns its CUDA stack (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), isolated from Arch's fast-moving system CUDA so `pacman -Syu` can't silently break CTranslate2's ABI. Gives a lockfile. Ruled out: Arch system CUDA (ABI churn), plain venv/pip (no lockfile, slower).
- **GPU spike is build-step zero.** Prove the 1060 can load+run turbo int8 on CUDA before architecting around a warm-in-VRAM model. Documented fallback if cuDNN fights: `device="cpu", compute_type="int8"` (slower but tolerable for short clips), debug GPU separately.

## Testing Decisions

This is a smoke harness, so the real acceptance is behavioural/manual (speak → observe), but the `Engine`/UI split creates clean seams:

- **Formatter (pure function):** unit-test directly — whitespace trim/collapse, empty/`no_speech` → dropped. No I/O, fast, deterministic. Highest-value automated test.
- **Engine transcribe seam:** a test that runs `Engine.transcribe(fixture_wav) -> text` against a short committed WAV of known speech and asserts the expected words appear. Marked to **skip when no CUDA/model available** (so it doesn't block CI/other machines); primarily a local confidence check.
- **Manual DoD check:** the Phase-1 acceptance criteria below, exercised live with the mic.
- Prior art: none (greenfield) — establish the convention: pure logic unit-tested, model/audio behind seams that can be faked or skipped.

**Phase 1 Definition of Done:** the loop prints accurate transcripts of my speech, prints nothing (or a quiet marker) on silence, and each transcription lands in **under ~1.5s wall time on GPU** once the model is warm. The latency clause is the proof the warm-model premise holds before Phase 2 is built on it.

## Steps

- [x] **Step 0 — GPU spike.** `scripts/gpu_spike.py` loads `large-v3-turbo` int8 on `device="cuda"`, transcribes the JFK fixture (verbatim-correct), prints timing + VRAM. GPU confirmed working (VRAM +1119 MiB, warm 0.938s/11s); no CPU fallback needed. Debugging sub-steps in `plan/step-gpu-spike.md` were not required.
- [x] **Step 1 — Project scaffold.** `uv` src-layout with `pyproject.toml` pinning `faster-whisper`, `sounddevice`, `numpy`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`; `uv sync` succeeds (Python 3.13, `ctranslate2==4.8.1`); `src/susurro/` package + `README` in place. `uv.lock` committed.
- [x] **Step 2 — Audio capture module.** `susurro.audio.record_window` records a fixed-duration 16kHz mono float32 window via callback, with device selection (`--device`, `list_input_devices`) + `PortAudioError` handling. `_WindowBuffer` + `load_wav` are hardware-free and unit-tested (7 tests); `sounddevice` is lazy-imported.
- [x] **Step 3 — Engine + formatter.** `susurro.engine.Engine.transcribe(audio) -> text` with the agreed defaults (`vad_filter=True`, `condition_on_previous_text=False`, `language="en"`, int8/cuda); model loaded once (warm) and reused. Pluggable `RuleBasedFormatter` (whitespace trim/collapse, drop empty/no-speech, no filler removal) with 9 passing unit tests; engine seam test (2, CUDA-gated).
- [x] **Step 4 — Smoke-harness loop.** `uv run susurro` loops capture→transcribe→format→print, prints a quiet `·` marker on silence, handles Ctrl-C cleanly. Warms CUDA before the first window. Non-interactive paths (`--help`, `--list-devices`) smoke-tested.
- [x] **Step 5 — Verify DoD.** Verified live by the user: `uv run susurro` produces accurate transcripts, stays quiet on silence, and lands warm results under the ~1.5s target (automated proxy: 0.938s). **Phase 1 DoD met.**

## Phase 2 — Plan (active)

Real system-wide hold-to-talk dictation on Hyprland, built on Phase 1's warm `Engine`: **hold a key → speak → release → clean text injected into the focused window**, fully local. The Step-6 spike (below) proved the trigger + injection mechanism live and set the decisions.

### Phase 2 Decisions

- **Trigger = Hyprland `bind`(press)/`bindr`(release) on a modifier-free dedicated key, + a daemon safety auto-stop timeout.** The spike proved release-driven start/stop works, but also that a **modifier chord (`SUPER,R`) drops the release event if the modifier is lifted first** → stuck "recording" state. A lone dedicated key's release always fires; a max-record timeout in the daemon guarantees a missed release can't wedge it. Final key is chosen at wiring time by quick hardware detection (`wev`/`hyprctl devices`); candidates: `Menu`, a spare Fn/`F13+`, or a **mouse side-button** (`mouse:275/276`, ideal for PTT). **evdev / `input`-group access is NOT needed** — the compositor is the trigger source (the plan's original Phase-2 permission blocker is moot).
- **Daemon owns the warm model; client is thin.** Model load + CUDA warm is amortized once at daemon start (Hyprland `exec-once`); per-utterance cost is just `transcribe`. The client is a trivial socket write, so release→text latency is dominated by inference, not process spawn — exactly the split the Phase-1 "daemon deferred until an external trigger forces it" decision predicted. Client must stay light (a fresh `python3` per call was noticeable in the spike); keep minimal-import or make resident if latency bites.
- **IPC = Unix stream socket, tiny line protocol (`start`/`stop`).** No framework. Socket at `$XDG_RUNTIME_DIR/susurro.sock` (per-user, tmpfs, auto-cleaned; unlink on daemon start/exit). Daemon is authoritative for state; single client at a time.
- **Injection = `wtype`** (Wayland virtual-keyboard protocol; proven in the spike, Hyprland-native, no uinput/root). Empty/no-speech transcript → no-op (don't type). **Clipboard+paste** (`wl-copy` + synthesized paste) kept as a documented fallback for long paragraphs / apps that drop fast synthetic keys — not the default (clobbers clipboard, per-app paste shortcut varies).
- **Variable-length capture reuses the Phase-1 callback model.** Stream opens on `start`, closes on `stop`; `_WindowBuffer` assembles arbitrary-length mono float32. No fixed duration; a max-duration cap ties into the safety timeout.
- **Cleanup stays pluggable.** Phase-1 `RuleBasedFormatter` is the default; an **ollama-backed LLM formatter** (~3B Q4, `ollama` is present) is an isolated, feature-flagged later step behind the existing `Formatter` seam — it does not block core hold-to-talk.
- **Daemon lifecycle = Hyprland `exec-once` autostart.** Warm before first use. Crash recovery = user restart for now; systemd-user supervision deferred (noted, not built).

### Phase 2 Steps

- [x] **Step 6 — Trigger+injection spike.** `scripts/trigger_spike.py`: Hyprland `bind`/`bindr` → Unix socket → `wtype`. Live-verified: release-driven start/stop + injection into the focused window both work; surfaced the modifier-release fragility that drives the trigger + timeout decisions. See `plan/log.md`.
- [x] **Step 7 — Variable-length capture.** Add start/stop recording to `susurro.audio` (a `Recorder` / `record_stream` reusing `_WindowBuffer` + the callback `InputStream`), returning arbitrary-length mono float32 with a max-duration cap. Unit-test the buffer assembly hardware-free (extends the Phase-1 `_WindowBuffer` tests); `sounddevice` stays lazy-imported.
- [x] **Step 8 — Daemon core.** `susurro.daemon`: warm `Engine` + `Recorder`, Unix-socket listener, idle↔recording state machine, safety auto-stop timeout; on `stop` → transcribe → format → inject. Clean socket lifecycle, single-flight. State machine unit-tested with fake recorder/engine via DI (like the formatter seam).
- [x] **Step 9 — Client + Hyprland trigger.** `susurro-ctl {start,stop}` entry point (low-latency socket write); detect + choose the modifier-free key; document the `bind`/`bindr` snippet and the `exec-once` autostart line.
- [x] **Step 10 — Injection module.** Extract + harden the spike's injection into `susurro.inject.inject(text)` via `wtype` (no-op on empty, surface `wtype` failures); document the clipboard-paste fallback. Unit-test command construction / empty-text no-op.
- [ ] **Step 11 — Wire end-to-end + live DoD.** Full hold→speak→release→text-in-focused-window with autostart. DoD: accurate transcript lands in the correct focused window, warm latency comparable to Phase 1 (~≤1.5s for short utterances), and a missed release cannot wedge the daemon (timeout verified).
- [ ] **Step 12 (optional) — LLM cleanup formatter.** ollama-backed `Formatter` behind the existing seam, feature-flagged; compare against rule-based on real dictation.

**Phase 2 DoD:** holding the trigger key, speaking, and releasing injects an accurate transcript into whatever window is focused — fully local, warm latency comparable to Phase 1, with no stuck-recording failure mode.

## Out of scope

- **Phase 3** (future, optional): cloud transcription/cleanup backend behind the same formatter/engine seams.
- Phase 2's trigger is the **compositor**, so evdev / `/dev/input` / `input`-group setup is **not** needed (proven in the Step-6 spike).
- **Streaming / partial results** — explicitly dropped, not a requirement in any phase.
- **Filler-word removal in Phase 1** — deferred to the Phase 2 LLM stage.
- **Multilingual support** — English-only by decision.
- **Toggle / always-listening trigger models.**
- **Vocabulary biasing** (`initial_prompt` with technical terms) — parked until a real term list exists.
