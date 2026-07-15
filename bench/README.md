# susurro-bench

An offline harness to answer one question: **which voice-to-text model is best for
susurro on *this* machine and *this* voice?** Judged on accuracy (WER/CER) and
transcribe latency (RTF).

Published benchmarks don't transfer, because they use other speakers, other GPUs,
and other precisions. This one records the owner reading a fixed set of scripts,
runs every available model over those recordings, and prints a comparable table.
The numbers are exclusive to the owner's hardware (NVIDIA GTX 1060 6GB, Pascal) and
voice, which is exactly what's needed to make the keep-or-switch call on the current
default (`large-v3-turbo`).

## Contenders

| id | runtime | notes |
|---|---|---|
| `fw-large-v3-turbo` | faster-whisper (int8/CUDA) | the current default / baseline |
| `fw-large-v3` | faster-whisper (int8/CUDA) | bigger, slower; is it more accurate? |
| `whispercpp-turbo` | whisper.cpp GGUF (pywhispercpp) | a different runtime for the same model |
| `parakeet-tdt-0.6b-v2` | NVIDIA NeMo | non-Whisper architecture, English-only |

Parakeet-v2 is English-only, which is why the whole benchmark is English-only: a
head-to-head needs one shared set. Portuguese/bilingual evaluation is out of scope.

## Install

The benchmark's runtimes live in optional extras, so the daemon's footprint and its
CUDA/CTranslate2 ABI stay untouched. Install only what you want to measure:

```bash
uv sync --extra bench                       # scoring (jiwer) + the two faster-whisper models
uv sync --extra bench --extra bench-whispercpp   # + whisper.cpp
uv sync --extra bench --extra bench-parakeet      # + Parakeet (pulls torch via NeMo)
```

A runtime you don't install is simply skipped (see `--list-models`); it never blocks
a run. Check what's wired and available:

```bash
uv run susurro-bench --list-models
```

## Workflow

### 1. Record yourself reading the scripts

```bash
uv run susurro-bench record            # walks every not-yet-recorded script
uv run susurro-bench record --only 05-numbers-and-dates   # redo one script
uv run susurro-bench record --redo     # re-record all of them
```

For each script it prints the text, waits for you to press Enter to **start**, then
Enter again to **stop**, and writes `bench/recordings/<id>.wav` (16kHz mono, via the
app's own `Recorder`). Recordings are personal voice data and are **gitignored**.
Already-recorded scripts are skipped unless you pass `--redo` or `--only`.

Read at a natural, steady pace. The scripts (`bench/scripts/*.txt`) span natural
prose, technical/code terms, numbers, and proper nouns.

### 2. Run the benchmark

```bash
uv run susurro-bench run                                  # every available model
uv run susurro-bench run --models fw-large-v3-turbo,fw-large-v3   # a subset
```

It pairs each `scripts/<id>.txt` (reference) with `recordings/<id>.wav` (audio) by
filename stem, builds each model once, and for every clip discards one warm-up
inference then times **N=3** runs. It scores each transcript, aggregates per model,
and writes a Markdown table to stdout and to `bench/results/results-<timestamp>.md`
(also gitignored).

## What the numbers mean

Per model, aggregated as the **median across clips** (robust to one bad clip):

| column | meaning | lower is better |
|---|---|---|
| **norm WER** | word error rate after normalization (lowercase, strip punctuation, collapse whitespace). The comparable **headline** number. | ✓ |
| **raw WER** | word error rate with case and punctuation preserved. Closest to what actually gets typed. | ✓ |
| **CER** | character error rate on the normalized text. A tiebreaker that de-noises word-boundary and tokenization quirks. | ✓ |
| **median latency (s)** | median warm transcribe time for a clip. What you actually wait on release. | ✓ |
| **median RTF** | real-time factor = transcribe seconds / audio seconds. Length-normalized, so it's comparable across clips of different lengths. RTF < 1 means faster than real time. | ✓ |
| **load (s)** | one-off model build time. Informational only — the daemon is warm in real use, so this is not part of per-utterance latency. | — |

**Timing method:** the first inference after a build compiles CUDA kernels and is
much slower, so it's discarded; the reported latency is the median of the following
three warm runs. This reflects the warm daemon, not a cold start.

**A note on numbers:** scoring uses `jiwer`'s transforms only — there is no
number-word normalization (that would pull in openai-whisper's torch stack). So `42`
vs `forty two` counts as an error. The scripts lean on forms that large-v3 renders
consistently (years, values ≥ 10, `percent` spelled out), but a model that spells a
number the reference wrote as a digit will take a small WER hit that isn't really a
mistake. Weigh CER and listen to the transcript before over-reading a WER gap that's
concentrated in the numbers script.

## Results & the decision

> **Status: pending the owner's recordings + the two heavy runtimes.**
> The head-to-head below needs `bench/recordings/` populated (your voice) and
> `whispercpp-turbo` + `parakeet-tdt-0.6b-v2` installed. Record, run, then paste the
> generated table here and fill in the call.

Fill this in after `susurro-bench run` over your recordings:

| model | norm WER | raw WER | CER | median latency (s) | median RTF | load (s) |
|---|---|---|---|---|---|---|
| fw-large-v3-turbo | | | | | | |
| fw-large-v3 | | | | | | |
| whispercpp-turbo | | | | | | |
| parakeet-tdt-0.6b-v2 | | | | | | |

**How to make the keep-or-switch call:**

- **Accuracy:** compare **norm WER** first (the headline). A difference under ~0.01
  on clean reads is usually noise; use **CER** as the tiebreaker and, for a close
  call, actually read the transcripts. Remember the number-word caveat above.
- **Speed:** compare **median RTF**. On this GPU every contender should be well under
  1.0; the question is how much headroom, since it's what you wait on each release.
- **Keep `large-v3-turbo`** unless a challenger is *both* clearly more accurate
  (meaningfully lower norm WER/CER, not just on the numbers script) *and* not
  painfully slower. A model that's a little more accurate but 2x the latency is not
  worth it for a hold-to-talk dictation tool.

### Pipeline-validation data point (not the decision basis)

The two faster-whisper models over the committed JFK fixture (`tests/fixtures/
jfk_16k_mono.wav`, ~11s), using the clip's transcript as the reference. This is a
single public clip, **not** the owner's script set, so treat it only as proof the
harness runs end-to-end on this hardware:

| model | norm WER | raw WER | CER | median latency (s) | median RTF | load (s) |
|---|---|---|---|---|---|---|
| fw-large-v3-turbo | 0.000 | 0.000 | 0.000 | 0.890 | 0.081 | 3.4 |
| fw-large-v3 | 0.000 | 0.000 | 0.000 | 1.485 | 0.135 | 139.1 |

Both nail this famous, clean read (0.0 WER), and turbo is ~1.7x faster per clip.
Whether that speed edge holds and whether accuracy separates on harder, owner-voiced
material is the point of running the full set.
