# Log — spec-6 preload on key-press

Append-only record of what's been done.

- 2026-07-07 — Spec written and approved. Option B (build-only preload) chosen over
  build+warm. Steps 1-3 agreed; Step 2 broken out into step-2-preload-on-start.md.
- [x] Step 1 — `LazyEngine.load()` added (build-only, idempotent, applies remembered language); `transcribe` now routes through it. 4 new tests green in test_lazy_engine.py.
- [x] Step 2 — `Daemon.start()` preloads on press via `_preload()` (managed + not loaded gate, best-effort/swallow+log); `_ManagedEngine` protocol declares `load()`; stale LazyEngine comment fixed; 5 daemon behaviours + fake `load()` added. Full suite 127 passed, ruff clean.
- [x] Step 3 — Manual verify on real daemon (CUDA + mic, idle-timeout 10s): after each idle-unload, `loading model` appears on the key-press (confirmed by user), so the build overlaps the hold. Preload confirmed working end-to-end.
