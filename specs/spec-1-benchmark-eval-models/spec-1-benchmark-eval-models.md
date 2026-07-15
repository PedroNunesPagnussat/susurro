# spec-1: Benchmark & evaluate voice-to-text models (speed + accuracy)

Issue: #1 — decide which model/runtime best fits susurro on the owner's hardware
and voice, judged on transcription accuracy (WER) and transcribe latency.

## Background

susurro is a warm-daemon dictation app. The transcription core is already
UI-agnostic and reusable offline, which is exactly what a benchmark needs:

- `Engine.transcribe(audio: np.ndarray) -> str` (`src/susurro/engine.py:56`) owns a
  warm faster-whisper model. A model is just a construction: `model_name`,
  `compute_type`, `device`, `language`, `beam_size`, `vad_filter`. So the two
  faster-whisper contenders (`large-v3-turbo`, `large-v3`) are two `Engine`s.
- `audio.load_wav(path) -> np.ndarray` (`src/susurro/audio.py:150`) loads 16-bit PCM
  WAV as mono float32. Its docstring already anticipates this work: "Feeds the Engine
  offline (without a mic) for tests and model evaluation."
- `Recorder` (`src/susurro/audio.py:63`) captures 16kHz mono float32 from the mic —
  reusable to capture the benchmark takes.
- Protocol style to mirror: `Formatter` (`src/susurro/formatter.py:20`) and
  `_ManagedEngine` (`src/susurro/daemon.py`) are `runtime_checkable` `Protocol`s. A
  `Transcriber` protocol follows the same pattern.
- Entrypoints are registered in `pyproject.toml:16` (`susurro`, `susurro-daemon`,
  `susurro-ctl`); a `susurro-bench` script fits the pattern.
- Deps are deliberately lean (`pyproject.toml:7`): faster-whisper + sounddevice +
  numpy + the CUDA wheels. The benchmark's extra runtimes must NOT land here.
- Prior art for tests: pure logic unit-tested with fakes (`tests/test_formatter.py`,
  `tests/test_lazy_engine.py`, `tests/test_daemon.py`); the real-model seam test skips
  without CUDA (`tests/test_engine.py:26`). The benchmark follows both.

Hardware context: NVIDIA GTX 1060 6GB (Pascal). Pascal has weak fp16 throughput, so a
runtime's real speed here is not predictable from spec sheets — which is the whole
point of measuring on-device.

## Problem Statement

The owner runs faster-whisper `large-v3-turbo` (int8, CUDA) and is happy with it, but
has no measured basis to know whether an alternative model or runtime would be
meaningfully more accurate or faster *on this specific hardware and voice*. Published
benchmarks don't transfer: they use other speakers, other GPUs, other precisions.

There is no way today to feed a fixed set of recordings through several models and get
back a comparable accuracy + latency table.

## Solution

A small offline benchmark harness, `susurro-bench`, that:

1. Holds a fixed set of Claude-authored English reference scripts (`bench/scripts/`).
2. Records the owner reading each script once (`susurro-bench record`), reusing the
   app's `Recorder`, into `bench/recordings/` (personal, gitignored).
3. Runs every available model over every recording (`susurro-bench run`), measuring
   transcribe latency (warm, load excluded) and scoring the transcript against the
   reference, then prints a Markdown comparison table.

Models are pluggable behind a `Transcriber` protocol. faster-whisper (`large-v3-turbo`,
`large-v3`) works out of the box via the existing `Engine`. whisper.cpp and
Parakeet-tdt-0.6b-v2 are lazy-import adapters gated behind an optional dependency group,
so an uninstalled runtime is skipped cleanly rather than blocking the whole run. The
benchmark's numbers are therefore exclusive to the owner's hardware and voice, which is
exactly what's needed to make the keep-or-switch call.

## Decisions

- **Reuse `Engine` + `load_wav`; don't re-implement transcription.** The faster-whisper
  contenders are two `Engine` constructions; the offline audio path already exists. The
  harness is orchestration + scoring + reporting around parts that are already tested.

- **`Transcriber` protocol as the common seam.** `runtime_checkable Protocol` with
  `name: str` and `transcribe(audio: np.ndarray) -> str`, mirroring `Formatter`. Every
  backend (faster-whisper, whisper.cpp, Parakeet) implements it; the runner is
  backend-blind. *Alternative ruled out:* per-backend `if/elif` in the runner — pushes
  runtime specifics into the orchestration and makes adding a model shotgun surgery.

- **Backends build their model warm in `__init__`** (like `Engine`), and expose a
  stable `name`. A model registry maps `id -> (factory, availability check)`; the runner
  builds only models that are both selected (`--models`) and importable. A missing heavy
  runtime is skipped with a message, never a crash. *Why:* the easy-first sequencing —
  faster-whisper always runs; the exotic runtimes join when installed.

- **Heavy deps isolated in `[project.optional-dependencies] bench`.** `jiwer` (scoring),
  plus `pywhispercpp` and the NeMo stack behind their adapters, live in an optional
  group installed with `uv sync --extra bench` (or a dedicated group). Core
  `dependencies` in `pyproject.toml:7` stay untouched, so the daemon's footprint and ABI
  are unaffected. *Alternative ruled out:* adding them to core deps — bloats the app and
  risks CTranslate2/CUDA ABI conflicts for zero runtime benefit.

- **No `openai-whisper` dependency for normalization.** It pulls torch and a large
  stack. `jiwer`'s own transforms (`ToLowerCase`, `RemovePunctuation`,
  `RemoveMultipleSpaces`, `Strip`) do the normalization. *Trade-off:* we forgo Whisper's
  number-word normalizer; scripts avoid ambiguous digit/word number forms and this
  sensitivity is documented.

- **WER: normalized headline + raw + CER reported together.** Normalized WER (lowercase,
  strip punctuation, collapse whitespace) is the comparable headline number. Raw WER
  (case + punctuation preserved) reflects what actually gets typed. CER is a
  normalization-robust secondary tiebreaker. All three per model. *Why all three:*
  normalized is comparable across models/literature; raw is closest to lived quality;
  CER de-noises word-boundary and tokenization quirks.

- **Latency: warm, load excluded, median of N=3 timed runs, RTF as headline.** For each
  (model, clip): one warm-up inference discarded (first call compiles CUDA kernels), then
  N=3 timed `transcribe` runs, report the median. Model load/build time is measured
  separately as informational, not in the per-clip latency (the daemon is warm in real
  use). The length-normalized headline is **RTF = transcribe_seconds / audio_seconds**;
  raw median latency is reported alongside. *Why RTF headline:* clips differ in length,
  so raw latency isn't comparable across clips; RTF is. Raw latency stays because it's
  what the owner actually waits.

- **All audio is 16kHz mono float32, guaranteed by the recorder — no resampling.**
  `Recorder`/`load_wav` produce 16kHz mono, which faster-whisper, whisper.cpp, and
  Parakeet all consume directly. One audio path for every backend.

- **Reference text = the script file's content; pair to a recording by filename stem.**
  `bench/scripts/<id>.txt` holds the reference; `bench/recordings/<id>.wav` is the take.
  No separate manifest — the shared stem is the join. *Alternative ruled out:* a
  manifest file mapping ids to text — extra indirection for no gain at this scale.

- **Scripts committed; recordings and results gitignored.** Scripts are shared,
  reviewable text. Recordings are the owner's voice (personal, binary) and results are
  regenerated output; both are gitignored. *Why:* the benchmark is explicitly personal
  ("exclusive to my use case"), so recordings aren't a shared artifact.

- **New `src/susurro/bench/` subpackage** (a slight departure from the otherwise flat
  module layout) because the benchmark has several distinct concerns — protocol +
  adapters, scoring, runner/report, CLI — and grouping them keeps the optional-runtime
  imports quarantined from the core package. Entrypoint `susurro-bench`.

- **English only.** All four models compete head-to-head on one English set. Parakeet-v2
  is English-only, so a Portuguese set would either handicap it or exclude it; the En↔Pt
  question is out of scope for this benchmark. *Alternatives ruled out:* a Portuguese
  subset for the Whisper models, or full bilingual — deferred, not part of the
  keep-or-switch decision being made now.

## Testing Decisions

- **Behaviour, not implementation.** Score a known (reference, hypothesis) pair and
  assert the expected WER/CER (e.g. one substitution in ten words → 0.1 normalized WER);
  assert the runner, given a fake transcriber with canned output over a real fixture WAV,
  produces a timed, scored, aggregated table without touching CUDA or a mic.
- **Seams tested:**
  - Scoring (`wer` module) — pure, unit-tested with hand-checked pairs. Prior art:
    `tests/test_formatter.py`.
  - Runner/report — unit-tested with a `FakeTranscriber` (canned text, optional small
    sleep to exercise median timing) over the committed `tests/fixtures/jfk_16k_mono.wav`.
    Prior art: `FakeManagedEngine`/`FakeClock` (`tests/test_daemon.py`), fake factory
    (`tests/test_lazy_engine.py`).
  - The pure WAV-writing helper behind `record` — unit-tested (round-trips a float32
    array through `load_wav`). Prior art: `tests/test_audio.py`.
- **Not unit-tested (real hardware):** the faster-whisper/whisper.cpp/Parakeet adapters
  running actual models, and mic capture in `record`. The faster-whisper adapter may get
  a `skipif`-CUDA seam test like `tests/test_engine.py:26`; the exotic backends are
  verified manually when installed. `uv run ruff check src tests` and `uv run pytest`
  stay green (heavy backends never imported at collection time).

## Steps

- [x] **Step 1 — Scaffold + faster-whisper backend.** Done when the `susurro-bench`
  entrypoint exists, a `Transcriber` protocol is defined, a faster-whisper adapter builds
  `large-v3-turbo` and `large-v3` via `Engine` behind the protocol, a model registry maps
  ids to factories + availability, and `jiwer` lives in a `bench` optional-dependency
  group (core `dependencies` unchanged; `uv run ruff` + `uv run pytest` green).
  Detail: [step-1-scaffold-fw-backend.md](step-1-scaffold-fw-backend.md).

- [x] **Step 2 — Reference scripts (Claude-authored).** Done when `bench/scripts/` holds
  ~8–10 English `.txt` scripts (~15–40s read each, ~1000+ words total) spanning natural
  prose, technical/code terms, numbers + punctuation, and proper nouns, committed; each
  file's content is its reference text, keyed by filename stem.

- [x] **Step 3 — Guided recording workflow.** Done when `susurro-bench record` walks the
  scripts in turn, shows each, and captures a 16kHz mono WAV to `bench/recordings/<id>.wav`
  (gitignored) via the existing `Recorder`, skips ids already recorded unless asked to
  redo, and can redo a single script by id.

- [x] **Step 4 — Scoring (WER/CER + normalization).** Done when a `wer` module returns
  normalized WER (headline), raw WER, and CER for a (reference, hypothesis) pair using
  `jiwer`, with normalization = lowercase + strip punctuation + collapse whitespace,
  unit-tested against hand-checked pairs.

- [x] **Step 5 — Runner + report.** Done when `susurro-bench run` transcribes every
  recording with every available+selected model (warm-up discarded, median of N=3 timed
  runs, RTF computed), scores each against its reference, and emits a Markdown table (per
  model: normalized WER, raw WER, CER, median latency, median RTF; plus informational
  load time) to stdout and `bench/results/` (gitignored); runner logic unit-tested with
  the JFK fixture + a `FakeTranscriber`; the faster-whisper models produce a full table
  end-to-end. Detail: [step-5-runner-report.md](step-5-runner-report.md).

- [x] **Step 6 — whisper.cpp backend.** Done when a lazy-import `WhisperCppTranscriber`
  (optional dep, e.g. `pywhispercpp`) runs `large-v3-turbo` GGUF through the `Transcriber`
  protocol and joins the results table when installed; when the runtime/model is absent it
  is skipped with a clear message, not a crash, and the rest of the run proceeds.

- [x] **Step 7 — Parakeet backend.** Done when a lazy-import `ParakeetTranscriber` (NeMo,
  optional dep) runs `parakeet-tdt-0.6b-v2` through the `Transcriber` protocol and joins
  the table when installed; absent → skipped cleanly like Step 6.

- [~] **Step 8 — Docs + the decision.** Done when the bench workflow is documented
  (install the `bench` extra → `record` → `run`, with the metric definitions) and a
  captured results summary compares all four models on WER + latency/RTF for the owner's
  voice/hardware, enough to make the keep-or-switch call on `large-v3-turbo`.
  *Status:* docs + metric definitions + a results template & reading guide done
  (`bench/README.md`); the captured **four-model** table and the final call are
  owner-pending — they need `bench/recordings/` (the owner's voice) and whisper.cpp +
  NeMo installed, so the owner runs `susurro-bench run` and pastes the table in.

## Out of scope

- Portuguese / bilingual benchmarking, and multilingual Parakeet-v3 (English-only chosen).
- Auto-installing the heavy backends — the owner installs the `bench` extra explicitly.
- Committing recordings to the repo (personal voice data; gitignored).
- Any change to core app `dependencies` or shipping the benchmark inside the daemon.
- Plots/GUI or statistical-significance testing beyond median (and optional p90).
- Number-word normalization beyond `jiwer` transforms (scripts avoid ambiguous forms).
- Tuning per-model decode params (beam size, VAD, quantization) as a sweep — each model
  runs one sensible configuration; a hyperparameter sweep is a separate effort.
