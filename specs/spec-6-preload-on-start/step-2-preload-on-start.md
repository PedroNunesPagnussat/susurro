# Step 2 — Preload on key-press in `Daemon.start()`

Wires `LazyEngine.load()` (Step 1) into the press path. Spans the engine protocol,
the state machine, the test fake, and the daemon tests, with load-order between them.

## Sub-steps

1. **Extend the `_ManagedEngine` protocol** (`daemon.py:56-68`): add `def load(self) -> None: ...`
   alongside `loaded`/`unload`/`transcribe`. Keeps the runtime-checkable contract in sync
   with what `start()` now calls.

2. **Preload inside `Daemon.start()`** (`daemon.py:140-156`): after `recorder.start()`,
   the `_recording`/timer bookkeeping, and `notify.recording(...)`, attempt a preload:
   if the engine is a managed engine and `not engine.loaded`, call `engine.load()`.
   Wrap it best-effort — on any exception, log (`self._log(...)`) and continue; do **not**
   re-raise, so a failed preload never aborts the capture. A small private helper
   (e.g. `_preload()`) keeps `start()` readable. Reuse the managed-engine detection the
   daemon already does (store an `isinstance(engine, _ManagedEngine)` flag at construction,
   or check inline) so a plain engine is a clean no-op.

3. **Fix the stale comment** on `LazyEngine` (`engine.py:78-80`): it says the first
   utterance after an unload "pays the full model-load + CUDA-warm cost — the accepted
   tradeoff." With preload, the build now overlaps the press; only the small first-inference
   warm remains on release. Update it to reflect that.

4. **Extend `FakeManagedEngine`** (`test_daemon.py:55-83`): add `load()` that sets
   `_loaded = True` and increments a `loads` counter, so tests can assert the press
   triggered a load and didn't double-load.

5. **Add daemon tests** (`test_daemon.py`) — the five behaviours:
   - *Press reloads an unloaded managed engine:* after `check_idle()` unloads, `start()`
     leaves `engine.loaded is True` **before** any `stop()`, and `loads == 1`.
   - *No double-load when already loaded:* on a warm engine, `start()` does not call
     `load()` (`loads == 0`); the existing warm engine stays loaded.
   - *Non-managed engine is a no-op:* with the plain `FakeEngine`, `start()` never attempts
     a load and behaves exactly as today (no attribute errors).
   - *Best-effort on failure:* a fake whose `load()` raises leaves the daemon `recording`
     (recorder started, `recording is True`) and a following `stop()` still transcribes and
     injects via the transcribe-path reload.
   - *Short press doesn't wedge:* `start()` immediately followed by `stop()` transcribes,
     injects, and returns to idle (`recording is False`).

## Ordering / notes

- Sub-step 1 before 2 (the protocol must declare `load()` before `start()` leans on it).
- Sub-step 4 before 5 (tests need the fake's `load()`).
- `start()`'s existing duplicate-press restart branch (`daemon.py:144-150`) is unchanged;
  preload sits after the recorder is (re)started, so a restart also benefits.
- Keep `_preload()` free of the fake-clock/timer logic — it only touches the engine.
