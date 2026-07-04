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

**Step 7 (variable-length capture):** DONE. Added `Recorder` + extended `_WindowBuffer` in
`susurro.audio` for Phase-2 hold-to-talk (arbitrary-length start/stop capture, mono float32,
`sounddevice` still lazy-imported).

- **`_WindowBuffer` cap:** added an optional `max_samples` ceiling. `add()` now *drops* blocks once
  full (bounds memory, not just a trim at the end) and flips a `capped` flag; `result()` defaults to
  the construction-time cap when called with no arg. `record_window` is unchanged (still passes an
  explicit `max_samples`), so Phase-1 behaviour is preserved.
- **`Recorder`:** `start()` opens the callback `InputStream` and begins accumulating into a fresh
  capped `_WindowBuffer`; `stop()` halts the stream *before* reading (race-free) and returns the
  captured audio trimmed to `max_duration_s` (default 30s). The callback closes over the local buffer
  (not `self._buf`), so a stray callback during `stop()` can't hit nulled state. Guards: `start()`
  raises if already recording, `stop()` raises if not. `max_duration_s` is the memory backstop; the
  daemon's safety timeout (Step 8) is the real stop.
- **Tests:** +4 (22 total). Hardware-free `_WindowBuffer` cap coverage (drops past cap, `capped`
  flag, `result` trim, `max_samples=None` keeps all) + two `Recorder` guards that don't touch
  PortAudio (`recording` False initially, `stop()`-without-`start()` raises). Full start/stop with a
  real mic is exercised live at Step 11; the daemon state machine gets a faked recorder at Step 8.
- **Suite:** 22 passed. `uvx ruff check` clean (ruff isn't a project dep — run it via `uvx ruff`,
  not `uv run ruff`).

**Step 8 (daemon core):** DONE. `susurro.daemon` — the warm always-on half of hold-to-talk. A
pure, dependency-injected `Daemon` state machine (idle<->recording) + a thin `serve()` socket shell,
plus a `susurro-daemon` console script.

- **`Daemon` (the seam):** `start()` / `stop()` / `check_timeout()` / `remaining()` / `abort()`.
  `recorder`, `engine`, `inject`, and `clock` are all constructor-injected, so the whole machine
  unit-tests with fakes + a fake clock — no mic, model, socket, or wall-clock. On `stop` (and on the
  safety timeout) it does transcribe -> format(engine-internal) -> inject; empty transcript => no
  inject.
- **Anti-wedge, three ways** (the Step-6 missed-release hazard): (1) safety auto-stop timeout
  (`max_record_s`, default 30s) stops+injects as if a real stop arrived; (2) `stop()` flips to idle
  *before* the fallible transcribe so a recorder/engine error can't leave it stuck recording; (3) a
  duplicate `start` (re-press after a lost release) discards the orphaned capture and restarts
  cleanly instead of raising. A late `stop` arriving after an auto-stop is a harmless no-op.
- **`serve()` socket shell:** Unix stream socket at `$XDG_RUNTIME_DIR/susurro.sock`, `start`/`stop`
  line protocol, single client, single-flight (transcription blocks the accept loop by design).
  Single-threaded, no locks: the safety timeout rides `accept()`'s socket timeout
  (`settimeout(remaining)`, floored to 0.05s so it can't flip to non-blocking and busy-spin). Clean
  lifecycle: unlink stale socket on bind, `abort()` (release mic) + unlink on shutdown/Ctrl-C. A
  failing command is caught, logged, and `abort()`ed so the daemon stays up.
- **Injection default:** a minimal inline `_wtype_inject` (mirrors the proven spike), explicitly
  marked for extraction into `susurro.inject` at Step 10. Tests never touch it (they inject a spy).
- **Recorder headroom:** the daemon builds `Recorder(max_duration_s = max_record + 5s)` so the
  daemon's timeout is always the authoritative stop and the recorder's buffer cap stays a pure
  memory backstop.
- **Tests:** +10 (32 total) covering happy path, empty-transcript no-inject, stop-without-start
  no-op, duplicate-start restart, idle-after-engine-error (no wedge), timeout fires/doesn't-fire/
  idle-noop, late-stop-after-auto-stop no-op, and `remaining()` countdown/clamp. Fakes:
  `FakeRecorder`/`FakeEngine`/`FakeClock` + an inject spy (DI, like the formatter seam).
- **Runnable:** `susurro-daemon --help` works without loading the model (argparse before Engine).
  Live serve() is exercised end-to-end at Step 11. Suite 32 passed; `uv run ruff check` clean (ruff
  now pinned in dev deps).

**Step 9 (client + Hyprland trigger):** DONE. `susurro.ctl` — the thin hold-to-talk client
(`susurro-ctl {start,stop}`) — plus a shared `susurro._ipc.socket_path`, and README docs for the
Hyprland `bind`/`bindr` + `exec-once` wiring. Did NOT touch the user's live Hyprland config (that's
the Step-11 live wiring, user's call).

- **Client is deliberately tiny + stdlib-only:** `import socket, sys` + `_ipc` (just `os`). No
  argparse, no numpy, no engine — so process startup stays cheap (the spike flagged a fresh
  `python3` per call as noticeable). `send(cmd)` opens the Unix socket, writes `start`/`stop`, closes.
  Exit codes: 0 sent, 1 daemon-not-running (`FileNotFoundError`/`ConnectionRefusedError`), 2 bad
  usage.
- **Shared socket path:** extracted `daemon._socket_path` into `susurro._ipc.socket_path()` (stdlib
  only) so daemon + client agree on `$XDG_RUNTIME_DIR/susurro.sock` without the client importing the
  heavy daemon module. Daemon updated to use it.
- **Console script:** `susurro-ctl = susurro.ctl:main` added to `pyproject.toml` (`susurro-daemon`
  landed in Step 8). Both resync clean.
- **Hardware probe (`hyprctl devices`):** box has a **Logitech MX Ergo trackball** (thumb
  side-buttons — ideal for PTT) + a Keychron K10 Pro. README recommends a mouse thumb button
  (`bind = , mouse:275, ...` / `bindr` on the same button), with `Menu` as the no-extra-hardware
  fallback, and tells the user to confirm the exact button code with `wev`. Autostart via
  `exec-once = .../.venv/bin/susurro-daemon` (venv console-script path, not `uv run`, to avoid
  working-dir/resolution surprises in Hyprland).
- **Tests:** +4 (36 total). Client tested against a *real* stdlib Unix socket (no daemon/model):
  delivers `start`/`stop` (asserted on the received bytes, thread joined to kill the accept race),
  `main` dispatches, daemon-absent -> rc 1, bad usage -> rc 2. `XDG_RUNTIME_DIR` pointed at a tmp dir
  so `_ipc.socket_path` is exercised for real.
- **Deferred to Step 11 (live):** picking/confirming the final key on the real hardware and the
  end-to-end `serve()`+`ctl` socket loop under Hyprland. serve() itself is intentionally not
  unit-tested (thin I/O shell; the state machine + client are covered). Suite 36 passed, ruff clean.

**Step 10 (injection module):** DONE. Extracted the daemon's inline `_wtype_inject` into
`susurro.inject.inject(text)` and hardened it; `daemon.main()` now passes `inject` as the DI'd
callable (tests still inject a spy, so the seam is unchanged).

- **What moved + what hardened:** the inline default was a faithful copy of the proven spike
  (`subprocess.run(["wtype", text])`, non-zero rc surfaced on stderr, never raises). The one added
  guard is the **empty/whitespace no-op** (`if not text.strip(): return`) so we never spawn wtype
  with nothing to type. The daemon already guards `if text:` before injecting; this is belt-and-braces
  at the seam (whitespace-only text is truthy, so the `.strip()` check is the real backstop).
- **Kept the never-crash contract:** scope is the documented non-zero-rc case (surface on stderr).
  A missing `wtype` binary still raises `FileNotFoundError` up into `serve()`'s broad handler (which
  logs + `abort()`s) — didn't widen the catch, matched `_wtype_inject` exactly to stay faithful.
- **Daemon cleanup:** dropped the now-unused `import subprocess` from `daemon.py`; `import inject`
  from `.inject`. `susurro-daemon --help` still works without loading the model.
- **Tests:** +5 (41 total). `test_inject.py` patches `susurro.inject.subprocess.run` (no real wtype):
  command construction is `["wtype", text]`, empty/whitespace (`""`, `"   "`, `"\n\t "`) is a no-op
  (no subprocess call), and a non-zero rc is surfaced on stderr without raising. TDD: wrote the test
  red (ModuleNotFoundError) before creating the module. Suite 41 passed; ruff clean.
- **README:** added an "Injection (and the clipboard fallback)" subsection documenting `wl-copy` +
  synthesized paste as a fallback-only (clobbers clipboard, per-app paste shortcut), plus `inject.py`
  in the layout + `test_inject.py` in the test list.
- **Deferred:** Step 11 (live end-to-end + DoD) still needs the user (Hyprland config edit + mic +
  GPU). Step 10 was fully in-scope with no hardware.

**Step 11 (wire end-to-end + live DoD):** IN PROGRESS — config wired live, DoD verification pending (user, needs mic/GPU).

- **Interaction model chosen: hold-to-talk on the `Menu` key** (user picked this over toggle). Toggle
  would need a new daemon `toggle` verb + client support (~15 lines + tests); NOT built — the existing
  start/stop hold model is used as-is (zero code change), matching the tested design.
- **Live config edits — OUTSIDE the repo, so they will NOT show in git:**
  - `~/.config/hypr/bindings.conf`:
    `bindd = , Menu, Dictate (hold to talk), exec, ~/dev/susurro/.venv/bin/susurro-ctl start`
    `bindr = , Menu, exec, ~/dev/susurro/.venv/bin/susurro-ctl stop`
    Verified registered via `hyprctl binds` (press -> start release=False; release -> stop release=True).
  - `~/.config/hypr/autostart.conf`:
    `exec-once = uwsm-app -- ~/dev/susurro/.venv/bin/susurro-daemon`
    (matches the existing `uwsm-app -- hyprsunset` convention; fires at next login).
  - `hyprctl reload` applied.
- **Daemon state:** a manual `uv run susurro-daemon` was already running (user-started), socket at
  `/run/user/1000/susurro.sock` — so it is testable now. The autostart (uwsm) daemon takes over at
  next login. The manual one may predate the Step-10 edit but is behaviorally identical (inject == old
  _wtype_inject for real text); restart only if you want the exact committed code.
- **REMAINING = the actual DoD (user's, on hardware):** hold Menu, speak, release -> accurate text in
  the focused window, warm latency ~<=1.5s, and a missed release cannot wedge (30s safety auto-stop
  verified). If Menu does not fire, confirm the keysym with `wev` and adjust the bind.
- **Open decision (from Step-10 review, NOT applied):** harden `inject.py` — add `subprocess.run(...,
  timeout=T)` + catch `TimeoutExpired`, and catch `FileNotFoundError` — so the now-public `inject`
  seam literally never raises and a stuck `wtype` can't permanently wedge the single-threaded accept
  loop (the last anti-wedge gap). Keep `T` generous (pathological-hang backstop, not a latency knob;
  a too-tight timeout SIGKILLs mid-type -> half-typed text). Would add 2 tests
  (`FileNotFoundError`/`TimeoutExpired` side_effects). Recommended for Step 11 live-hardening.
