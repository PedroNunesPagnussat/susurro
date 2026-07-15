# Step 1 — Scaffold + faster-whisper backend

Stand up the `src/susurro/bench/` subpackage, the `Transcriber` seam, the
faster-whisper adapter, the model registry, and the `bench` optional-dependency group.
This is the skeleton every later step hangs off; no scoring or recording yet.

## Done when

The `susurro-bench` entrypoint exists, a `Transcriber` protocol is defined, a
faster-whisper adapter builds `large-v3-turbo` and `large-v3` via `Engine` behind the
protocol, a model registry maps ids to factories + availability, and `jiwer` lives in a
`bench` optional-dependency group with core `dependencies` unchanged and `uv run ruff` +
`uv run pytest` green.

## Sub-steps

- [ ] **Create the subpackage.** `src/susurro/bench/__init__.py` and the module files
  the later steps fill in (`transcriber.py`, `registry.py`, `cli.py`; `wer.py`,
  `runner.py` land in Steps 4–5). Keep every heavy/optional import lazy so
  `import susurro.bench` stays cheap and collection-safe.

- [ ] **Define the `Transcriber` protocol** in `transcriber.py`: `runtime_checkable
  Protocol` with `name: str` and `transcribe(audio: np.ndarray) -> str`, mirroring
  `Formatter` (`src/susurro/formatter.py:20`). Document that `audio` is 16kHz mono
  float32 (the guaranteed harness format).

- [ ] **faster-whisper adapter** (`FasterWhisperTranscriber`): wraps an `Engine`
  (`src/susurro/engine.py:22`) built with a given model name + `compute_type="int8"`,
  `device="cuda"`; sets `name` (e.g. `fw-large-v3-turbo`). Its `transcribe` delegates to
  `Engine.transcribe`. Both faster-whisper contenders differ only by model name and both
  run int8/CUDA, so turbo-vs-large isolates the model, not the precision.

- [ ] **Model registry** (`registry.py`): an ordered mapping `id -> ModelSpec`, where a
  `ModelSpec` carries a human label, a `build()` factory returning a `Transcriber`, and an
  `available()` predicate (can the backend import / is the model present?). Register the
  two faster-whisper ids now; leave documented registry slots for `whispercpp-turbo`
  (Step 6) and `parakeet-tdt-0.6b-v2` (Step 7). faster-whisper `available()` is true when
  `faster_whisper` imports (it's a core dep).

- [ ] **CLI skeleton** (`cli.py` + `susurro-bench` in `pyproject.toml:16`
  `[project.scripts]`): argparse with `record` and `run` subcommands (bodies land in
  Steps 3 and 5) and a shared `--models` selector that filters the registry (default: all
  available). Reuse `_cli.py` conventions where they fit; a `--list-models` flag prints
  registry ids + availability and exits — enough to prove the wiring in Step 1.

- [ ] **Optional dependency group.** Add `[project.optional-dependencies]` `bench = [
  "jiwer>=3" ]` (pin later backends' deps here in Steps 6–7). Confirm core
  `dependencies` (`pyproject.toml:7`) are untouched and `uv sync --extra bench` resolves.

## Verify

- `uv run susurro-bench --list-models` lists the two faster-whisper ids as available.
- `uv run ruff check src tests` and `uv run pytest` are green (no heavy import pulled in
  at collection).
- `grep` confirms core `dependencies` unchanged; `jiwer` only under the `bench` extra.

## Notes / risks

- Keep `cli.py` import-light: importing it must not import `faster_whisper`, NeMo, or
  `pywhispercpp`. Build a `Transcriber` only when a run actually selects that model.
- `--list-models` availability check must not construct models (no CUDA/model download) —
  just probe importability.
