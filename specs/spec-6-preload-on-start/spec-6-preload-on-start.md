# spec-6: Preload the model on key-press, not just on transcribe

Issue: #6 (enhancement) — follow-up to #5 (idle-unload).

## Background

The daemon (`src/susurro/daemon.py`) is a pure, dependency-injected idle<->recording
state machine wrapped by a single-threaded socket loop (`serve`). It holds a
`LazyEngine` (`src/susurro/engine.py`) so an idle daemon can drop the Whisper model
and free VRAM, rebuilding it lazily.

Today the reload is lazy-*on-transcribe*:

- `main()` warms the model once at startup with a throwaway `engine.transcribe(np.zeros(...))`
  (`daemon.py:384`), paying build + first-inference warm up front.
- After an idle-unload (`Daemon.check_idle` → `LazyEngine.unload`), the model is gone.
- The next `LazyEngine.transcribe` rebuilds it (`engine.py:109-116`). That call only
  happens inside `Daemon.stop()` (`daemon.py:172`), i.e. on the key *release*.

So the first utterance after an unload pays the full model-load cost **after** the
user has already stopped speaking, and they wait with nothing happening.

Relevant seams and prior art:
- `LazyEngine` load/unload lifecycle: `engine.py:72-125`; unit-tested with a fake
  factory in `tests/test_lazy_engine.py`.
- `_ManagedEngine` protocol (runtime-checkable: `loaded`/`unload`/`transcribe`):
  `daemon.py:56-68`. The daemon detects a managed engine via `isinstance` to decide
  whether idle-unload is even possible (`daemon.py:111`).
- `Daemon.start()`: `daemon.py:140-156` — starts the recorder, flips to recording,
  posts the persistent recording toast.
- Daemon tests use `FakeManagedEngine` + `FakeClock` (`tests/test_daemon.py:55-93`),
  no CUDA/mic/socket.

## Problem Statement

After the model has been idle-unloaded, the first dictation is slow in the worst
possible place: the user finishes speaking, releases the key, and only then does the
multi-second model reload begin. The wait lands entirely on the release, when they're
expecting their text.

## Solution

On the key *press* (`Daemon.start()`), if the model is currently unloaded, start
rebuilding it immediately so the load overlaps with the user speaking. Capture starts
first (on the sounddevice callback thread), so audio is still recorded while the model
loads. By the time the user releases, the model is already built and `stop()` just
transcribes.

`LazyEngine` gains a `load()` entrypoint (build-only) that `Daemon.start()` calls when
the engine is managed and unloaded. The load is synchronous and best-effort.

## Decisions

- **Synchronous (blocking) load, no background thread.** `load()` runs inline in
  `start()`, blocking the single-threaded accept loop for the seconds of the build.
  Capture is unaffected because `recorder.start()` runs first and records on PortAudio's
  callback thread; a `stop`/`start` that arrives mid-load just queues until the load
  returns. The safety auto-stop is 60s, far longer than a load, so a blocked loop can't
  wedge. *Alternative ruled out:* background-thread the load to keep the loop responsive.
  It adds the first threading + locks to an intentionally lock-free daemon for no real
  benefit at hold-to-talk timescales (issue #6 calls it "probably overkill").

- **Ordering in `start()`: recorder + toast, then load.** `recorder.start()` and
  `notify.recording()` happen before the load so capture is live and the "Recording"
  toast is already on screen while the model builds behind it.

- **Build-only load (Option B), not build+warm.** `load()` constructs the model
  (`WhisperModel(...)`, the dominant multi-second cost) but does not run a warm
  inference. The smaller first-inference CUDA warm still lands on the release transcribe.
  *Why:* the build dominates, so preloading it captures nearly all the win; and the
  startup warm runs on silence with `vad_filter=True`, which VAD likely skips, so a
  build+warm preload's extra benefit is uncertain. *Alternative ruled out:* Option A
  (build + a throwaway inference in `load()`, reused at startup) — more code and a
  dummy-audio buffer inside `LazyEngine` for a benefit that may not materialize.

- **Best-effort preload.** If `load()` raises (e.g. CUDA OOM), `start()` logs it and
  keeps recording; the real `stop()` → `transcribe` retries and surfaces the error
  through the existing path. Preload is a pure optimization and must never make a press
  worse than today.

- **Gate on "managed engine AND not loaded", not on `idle_timeout_s`.** The model only
  ever becomes unloaded via idle-unload, so `not loaded` is the precise trigger; a plain
  (non-managed) engine, or an already-loaded one, is a no-op. Mirrors how the daemon
  already uses `isinstance(engine, _ManagedEngine)` to gate idle-unload.

- **Startup warm path unchanged.** `main()` keeps `engine.transcribe(np.zeros(...))` for
  the initial warm (it builds + warms). Only the post-idle press path uses `load()`.

- **API name `load()`** on `LazyEngine`, mirroring the existing `unload()`; added to the
  `_ManagedEngine` protocol.

## Testing Decisions

- **Behaviour, not implementation.** Assert the *observable* effect: after a press on an
  unloaded managed engine, the engine is loaded *before* `stop()` runs (the load moved to
  the press); an already-loaded engine isn't loaded again; a non-managed engine never
  attempts a load; a failing `load()` leaves the daemon recording and a following `stop()`
  still transcribes; a short press (start then immediate stop) still transcribes and
  returns to idle.
- **Seams:** `LazyEngine` via a fake factory (`test_lazy_engine.py`), and `Daemon` via
  `FakeManagedEngine` + `FakeClock` (`test_daemon.py`) — no CUDA/mic/socket, same as #5.
- **Prior art to follow:** `_factory_spy`/`FakeInner` (`test_lazy_engine.py:12-37`) for
  the load lifecycle; `FakeManagedEngine` (`test_daemon.py:55-83`) extended with a
  `load()` that flips `loaded` and counts calls.
- **Real hardware** (CUDA + mic) isn't unit-tested; Step 3 covers it with a manual verify.

## Steps

- [x] **Step 1 — Add `LazyEngine.load()` (build-only).** Done when `LazyEngine` exposes
  `load()` that builds the inner engine on first call (applying the remembered language)
  and is a no-op when already loaded, with `test_lazy_engine.py` covering: builds exactly
  once, idempotent when already loaded, remembered language reaches the fresh engine, and
  `load()` followed by `transcribe()` doesn't rebuild.

- [x] **Step 2 — Preload on key-press in `Daemon.start()`.** Done when `Daemon.start()`,
  after `recorder.start()` and the recording toast, calls `engine.load()` when the engine
  is managed and unloaded (best-effort: swallow + log a load failure without wedging the
  press); the `_ManagedEngine` protocol declares `load()`; the stale
  "first utterance after an unload pays the full cost" comment on `LazyEngine` is
  corrected; and `test_daemon.py` asserts the five behaviours listed in Testing Decisions.
  Detail: [step-2-preload-on-start.md](step-2-preload-on-start.md).

- [x] **Step 3 — Verify end-to-end on the real daemon.** Done when, with a low
  `idle_timeout_s`, the daemon log shows the reload beginning on the key-*press* (not the
  release), and a real dictation after an idle-unload injects text with the load
  overlapped by the hold. Manual `/verify` — CUDA + mic aren't in the unit suite.

## Out of scope

- Background-threading the load (rejected above).
- Warming the model during preload (Option A) — build-only is the chosen depth.
- A config flag to enable/disable preload — it's a pure best-effort optimization, no knob
  needed.
- Changing the startup warm path.
- Preloading speculatively before a press (e.g. on a timer) — preload is tied to the press.
