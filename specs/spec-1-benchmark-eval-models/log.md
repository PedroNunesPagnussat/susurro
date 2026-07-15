# Log — spec-1 benchmark & evaluate voice-to-text models

Append-only record of what's been done.

- 2026-07-08 — Spec written and approved. Decisions locked with the owner: English-only
  benchmark; WER reported as normalized headline + raw + CER; easy-first runtime
  sequencing (common `Transcriber` protocol + `bench` optional-dep group; faster-whisper
  first, then whisper.cpp, then Parakeet as separate deferrable steps). Latency = warm,
  load excluded, median of N=3, RTF headline. Steps 1–8 agreed; Steps 1 and 5 broken out
  into step detail files.
- [x] Step 1 — Scaffold + faster-whisper backend. `src/susurro/bench/` subpackage with
  `Transcriber` protocol, `FasterWhisperTranscriber` (via `Engine`, int8/CUDA), ordered
  model registry (import-probe availability, no model build), `susurro-bench` entrypoint +
  CLI skeleton (`--list-models`, `record`/`run` subparsers, `--models` selector). `jiwer>=3`
  under `[project.optional-dependencies] bench`; core deps unchanged. 173 tests green (+13),
  ruff clean; fw adapter seam test transcribes JFK on CUDA.
- [x] Step 2 — Reference scripts. 10 English `.txt` scripts in `bench/scripts/` (1023 words
  total), spread ~23-47s read across natural prose, technical/code terms, systems/GPU,
  numbers+punctuation, and proper nouns (places + people/brands). Length spread is
  deliberate so RTF has short and long clips. `.gitignore` ignores `bench/recordings/` +
  `bench/results/`; scripts are committed.
- [x] Step 3 — Guided recording workflow. `susurro-bench record` walks scripts, shows each,
  captures 16kHz mono WAV to `bench/recordings/<id>.wav` via the app's `Recorder` (Enter to
  start / Enter to stop); skips already-recorded ids, `--redo` re-records all, `--only <id>`
  redoes one. Pure seams unit-tested: `save_wav` round-trips through `load_wav` (incl.
  clipping) and `plan_recordings` skip/redo/only logic (11 tests). Mic capture is
  owner-verified. `--only <bad id>` fails loud (exit 2) before touching a mic.
- [x] Step 4 — Scoring (WER/CER). `bench/wer.py` `score(ref, hyp) -> Score(norm_wer, raw_wer,
  cer)` via jiwer. Raw WER = jiwer default (case+punct kept); normalized WER + CER share a
  lowercase/strip-punctuation/collapse-whitespace prefix (no number-word normalization). 6
  hand-checked unit tests (identical→0, 1 sub/10→0.1, case/punct-only→norm 0/raw 1.0,
  CER finer than WER, empty hyp→1.0). Guarded with importorskip so core-only pytest stays green.
- [x] Step 5 — Runner + report. `bench/runner.py`: `discover_clips` pairs script↔recording by
  stem (warns on missing/orphan); `run_models` builds each spec once (measures load), discards
  a warm-up, times median of N=3 via injected timer, computes RTF, scores via Step-4 wer, and
  aggregates per-model as the median across clips; `format_report` renders a Markdown table
  (baseline first, build-failed models omitted+noted) with a clip-count/total-audio header;
  `run_command` writes `bench/results/results-<ts>.md` + stdout. 9 unit tests (FakeTranscriber
  + FakeClock over JFK fixture: warm-up discard, median, RTF, real scoring, resilience). Full
  suite 199 green. Real-hardware: both fw models produce a two-row table end-to-end, RTF 0.081
  (turbo) / 0.135 (large-v3) on CUDA; run_command writes the timestamped results file.
- [x] Step 6 — whisper.cpp backend. Lazy-import `WhisperCppTranscriber` (pywhispercpp,
  `large-v3-turbo` GGUF, 16kHz mono float32, RuleBasedFormatter for parity) behind the
  `Transcriber` protocol; registered as `whispercpp-turbo` with a pywhispercpp import-probe
  availability. `pywhispercpp>=1.2` pinned in a separate `bench-whispercpp` extra (keeps
  `bench` light); resolves via `uv lock` (1.5.0). Absent runtime → `--list-models` shows
  unavailable and the runner skips it cleanly (import-probe test flips availability both ways).
  Real GGUF run is owner-verified-when-installed.
- [x] Step 7 — Parakeet backend. Lazy-import `ParakeetTranscriber` (NeMo,
  `nvidia/parakeet-tdt-0.6b-v2`, moved to CUDA, English-only, version-robust transcript
  extraction) behind the `Transcriber` protocol; registered as `parakeet-tdt-0.6b-v2` with a
  `nemo` import-probe. `nemo-toolkit[asr]>=2.0` pinned in a separate `bench-parakeet` extra;
  `uv lock` resolves it on py3.13 (torch stays out of the active env). Absent → `--list-models`
  unavailable + clean runner skip. Real NeMo run is owner-verified-when-installed. Core
  `dependencies` still the original 5; all four models registered.
- [~] Step 8 — Docs done; the 4-model decision is owner-pending. `bench/README.md` documents
  install (the three extras) → `record` → `run`, the metric definitions (norm/raw WER, CER,
  latency, RTF, load), the timing method, and the number-word/normalization sensitivity. Main
  README gained a Benchmarking section + updated Layout. Results summary is a template + a
  how-to-read-it keep-or-switch guide, with the real fw-only two-row JFK-fixture table as a
  labeled pipeline-validation data point. The captured **four-model** comparison and the final
  keep-or-switch call require the owner's voice recordings + whisper.cpp/NeMo installed, so
  they're left for the owner to run and paste in — not something reproducible in this session.
