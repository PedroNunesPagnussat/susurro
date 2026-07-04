# susurro — Work Log

Append-only record of what's been done. Newest at the bottom.

## 2026-07-03 — Phase 1

**Step 1 (scaffold):** `uv` project, src layout, `pyproject.toml` pinning `faster-whisper`,
`sounddevice`, `numpy`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`. `uv sync` OK on Python 3.13
→ `ctranslate2==4.8.1`, `faster-whisper==1.2.1`, cuDNN 9.24, cuBLAS 12.9. System `libportaudio`
present.

**Step 0 (GPU spike):** PASS on the GTX 1060.
- `large-v3-turbo` int8 on `device="cuda"` — venv-local cuBLAS/cuDNN wheels preloaded via
  `susurro._cuda.preload_cuda_libs()` (LD_LIBRARY_PATH is unreliable in-process; ctypes preload
  instead).
- VRAM: 916 → 2035 MiB (**delta 1119 MiB**), leaves ample headroom on 6GB.
- Transcript of the JFK fixture is verbatim-correct.
- **Warm transcribe: 0.938s for 11.0s of audio** → well under the 1.5s DoD (a 3s window is
  cheaper still). Warm-model premise holds; no CPU fallback needed.
- Fixture: `tests/fixtures/jfk_16k_mono.wav` (canonical public-domain JFK clip, ffmpeg → 16kHz
  mono s16, 11s).

**Step 2 (audio capture):** `susurro.audio` — `record_window()` (sounddevice/PortAudio, 16kHz
mono float32, callback-based), `_WindowBuffer` (hardware-free block assembly), device helpers,
`load_wav()`. `sounddevice` lazy-imported so the module imports without an audio server. 7 unit
tests (buffer + WAV load), no hardware needed.

**Step 3 (engine + formatter):** `susurro.formatter` (`Formatter` protocol + `RuleBasedFormatter`:
whitespace trim/collapse, drop empty/no-speech, no filler removal) — 9 unit tests, TDD red→green.
`susurro.engine.Engine` — warm model loaded once, `transcribe(np.ndarray)->str` with the agreed
defaults (`language="en"`, `vad_filter=True`, `condition_on_previous_text=False`, `beam_size=5`,
int8/cuda), formatter applied, pluggable formatter DI. Engine seam test (2, CUDA-gated): JFK words
present, silence→"". 6.4s.

**Step 4 (harness loop):** `susurro.__main__` — argparse (`--duration/--device/--model/--cpu/
--list-devices`), warm-up pass, capture→transcribe→format→print loop, quiet-marker (`·`) on
silence, clean Ctrl-C. Non-interactive paths smoke-tested (`--help`, `--list-devices` → sees the
fifine USB mic at idx 7).

**Suite:** 18 passed (9 formatter + 7 audio + 2 engine). All modules import clean.

**Step 5 (verify DoD):** DONE — verified live by the user. `uv run susurro` gives accurate
transcripts, quiet on silence, warm results under ~1.5s (automated proxy: 0.938s). **Phase 1 DoD
met; the mic → local Whisper GPU → text pipeline works and the warm-model premise holds.**

**Git + review:** repo initialised (`master`), initial commit `a6c778c`. `/code-review` (ruff +
manual smell pass): no correctness/security bugs. Fixed — removed unused `pytest` import
(`test_audio.py`), used `SAMPLE_RATE` instead of a magic `16_000` in the harness warmup. Ruff clean,
18 tests green after fixes. Remaining notes (device label vs. capture default, uncaught mid-loop
`record_window` error) left as accepted for a Phase-1 smoke tool.

## 2026-07-03 — Phase 2

**Step 6 (trigger + injection spike):** PASS, live-verified by the user. `scripts/trigger_spike.py`
(stdlib only, no audio/model/venv) — a minimal daemon+client over a Unix socket
(`$XDG_RUNTIME_DIR/susurro-spike.sock`) driven by Hyprland `bind`(press)/`bindr`(release), injecting
a marker via `wtype`. Purpose: de-risk the two Phase-2 unknowns before building the real daemon.

- **Trigger works:** press→`START`, release→`STOP`, hold duration measured accurately
  (`held=2.561s` etc. from `time.perf_counter`).
- **Injection works:** `wtype` typed `[susurro spike: held 2.56s] ` straight into the focused window
  (Wayland virtual-keyboard protocol — no uinput/root/`input` group). The plan.md `input`-group
  blocker is confirmed **moot**: trigger comes from the compositor, not evdev.
- **Modifier-held-during-injection did NOT bite:** feared SUPER+letter WM-bind collisions didn't
  fire, because the ~30ms client/socket startup let the user finish releasing SUPER before `wtype`
  typed. Clean marker landed.
- **Real finding — missed release:** with a modifier chord (`SUPER, R`), `bindr` is **skipped if
  SUPER is released before R** → an orphaned `START` with no `STOP` (daemon logged "START while
  already recording"). A stuck-recording state is a genuine hazard. → Drives Phase-2 decisions:
  **modifier-free dedicated key** (release always fires; final key TBD by hardware) + a **daemon
  safety auto-stop timeout** as a backstop.
- **Tooling on the box:** Hyprland 0.55.2 (Omarchy), `wtype` + `wl-clipboard` present, `ydotool`
  missing and `/dev/uinput` root-only (so `wtype` is the path anyway), `ollama` present (for the
  later LLM formatter). Client latency was a fresh `python3` start per call — note for the real
  client (keep it light or resident).
- **Teardown:** spike binds removed from `~/.config/hypr/bindings.conf` + `hyprctl reload`, daemon
  stopped, socket removed. `scripts/trigger_spike.py` kept as a reference artifact (like
  `gpu_spike.py`).
