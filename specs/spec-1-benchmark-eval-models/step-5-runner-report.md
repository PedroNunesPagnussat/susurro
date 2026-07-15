# Step 5 — Runner + report

The orchestration: for every selected+available model and every recording, time the warm
transcribe, score it, aggregate, and print/write a Markdown comparison table. Depends on
Step 4 (scoring) and Step 1 (protocol + registry); consumes recordings from Step 3.

## Done when

`susurro-bench run` transcribes every recording with every available+selected model
(warm-up discarded, median of N=3 timed runs, RTF computed), scores each against its
reference, and emits a Markdown table (per model: normalized WER, raw WER, CER, median
latency, median RTF; plus informational load time) to stdout and `bench/results/`
(gitignored); the runner logic is unit-tested with the JFK fixture + a `FakeTranscriber`,
and the faster-whisper models produce a full table end-to-end.

## Sub-steps

- [ ] **Discover the eval set.** Pair `bench/scripts/<id>.txt` (reference) with
  `bench/recordings/<id>.wav` (audio) by shared stem. A script with no recording is
  reported as missing (skip, warn); an orphan recording warns. Load audio via
  `audio.load_wav` (`src/susurro/audio.py:150`).

- [ ] **Per-(model, clip) measurement.** Build each model once (record load/build seconds
  as informational). For each clip: one warm-up `transcribe` discarded, then N=3 timed
  runs via `time.perf_counter`; keep the median transcribe time. Capture the transcript
  from one run for scoring. RTF = median_transcribe_s / (len(audio) / sample_rate).

- [ ] **Score.** Run each transcript through the Step 4 `wer` module against the paired
  reference → normalized WER, raw WER, CER. Aggregate per model across clips (mean or
  median WER/CER; document which — median is robust to one bad clip).

- [ ] **Report.** Emit a Markdown table sorted with the baseline (`fw-large-v3-turbo`)
  first: columns model | norm WER | raw WER | CER | median latency (s) | median RTF | load
  (s). Print to stdout and write `bench/results/results-<timestamp>.md`. Include the run's
  clip count + total audio seconds in a header line for context.

- [ ] **Resilience.** A model that fails to build (missing runtime/model) is skipped with
  a message and omitted from the table; the rest of the run proceeds. A single clip that
  errors on one model is recorded as a failure for that cell, not a whole-run abort.

## Testing (this step)

- **`FakeTranscriber`**: canned transcript per input + an optional tiny `sleep` so the
  median-of-N timing path is exercised deterministically without a real model. Prior art:
  `FakeManagedEngine` (`tests/test_daemon.py`), fake factory (`tests/test_lazy_engine.py`).
- Drive the runner over the committed `tests/fixtures/jfk_16k_mono.wav` with a fake
  reference, asserting: it pairs script↔recording, discards the warm-up, computes a median
  from N runs, produces WER/CER via the real scorer, and formats a table row. No CUDA, no
  mic.
- Assert resilience: a model whose factory raises is skipped and the others still report.

## Verify (real hardware)

- With recordings present, `uv run susurro-bench run --models fw-large-v3-turbo,fw-large-v3`
  prints a two-row table with plausible WER (< ~0.15 normalized on clean reads) and RTF < 1
  on CUDA, and writes the timestamped results file.

## Notes / risks

- Warm-up matters: the first CUDA inference compiles kernels and is much slower; excluding
  it is what makes the numbers reflect the warm daemon. Make the discard explicit/visible.
- Decide mean-vs-median aggregation once and state it in the report header so the number
  isn't ambiguous.
- Keep transcript capture and timing separate: capture the transcript from a run that is
  also timed (don't run a 4th untimed call), but score only once per (model, clip).
