"""`susurro-bench run` — the orchestration: for every selected+available model and
every recording, time the warm transcribe, score it, aggregate, and render a
Markdown comparison table.

Dependency-injected like the daemon so the whole loop unit-tests with a
`FakeTranscriber` + a fake clock (no CUDA, no mic): `run_models` takes the clips,
the model specs, the timer, and N. Only `run_command` touches the registry, the
real recordings, and the filesystem.

Timing method (see spec Decisions): one warm-up `transcribe` is discarded (the
first CUDA inference compiles kernels), then N timed runs are measured with the
injected timer and the **median** kept. RTF = median transcribe seconds / audio
seconds, so clips of different lengths stay comparable. Model build/load time is
measured separately as informational — the daemon is warm in real use. Per-model
accuracy is the **median** of the per-clip scores (robust to one bad clip).
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE, load_wav
from .registry import ModelSpec
from .wer import Score, score


@dataclass
class Clip:
    """One eval item: a script id, its reference text, and the recorded audio."""

    id: str
    reference: str
    audio: np.ndarray


@dataclass
class CellResult:
    """One (model, clip) measurement. `error` set (and the rest None) if it failed."""

    clip_id: str
    transcript: str | None = None
    score: Score | None = None
    median_latency_s: float | None = None
    rtf: float | None = None
    error: str | None = None


@dataclass
class ModelResult:
    """A model's per-clip cells plus the per-model aggregates. `build_error` set (and
    aggregates None, cells empty) when the backend couldn't even be built."""

    model_id: str
    label: str
    load_s: float | None = None
    cells: list[CellResult] = field(default_factory=list)
    norm_wer: float | None = None
    raw_wer: float | None = None
    cer: float | None = None
    median_latency_s: float | None = None
    median_rtf: float | None = None
    build_error: str | None = None


@dataclass
class RunResult:
    clips: list[Clip]
    models: list[ModelResult]
    sample_rate: int
    timed_runs: int


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def discover_clips(scripts_dir: Path, recordings_dir: Path) -> tuple[list[Clip], list[str]]:
    """Pair `scripts/<id>.txt` (reference) with `recordings/<id>.wav` (audio) by
    shared stem, in script order. Returns `(clips, warnings)`: a script with no
    recording is skipped with a warning, and an orphan recording (no script) warns
    too — neither aborts the run."""
    from .recording import recorded_ids, script_ids

    ids = script_ids(scripts_dir)
    recorded = recorded_ids(recordings_dir)
    clips: list[Clip] = []
    warnings: list[str] = []
    for id_ in ids:
        wav = recordings_dir / f"{id_}.wav"
        if not wav.exists():
            warnings.append(f"no recording for script '{id_}' — skipped (record it first)")
            continue
        reference = (scripts_dir / f"{id_}.txt").read_text().strip()
        clips.append(Clip(id=id_, reference=reference, audio=load_wav(wav)))
    for orphan in sorted(recorded - set(ids)):
        warnings.append(f"recording '{orphan}' has no matching script — ignored")
    return clips, warnings


def _measure_clip(
    backend,
    clip: Clip,
    *,
    timed_runs: int,
    warmup: bool,
    timer: Callable[[], float],
    sample_rate: int,
    log: Callable[[str], None],
) -> CellResult:
    """Warm-up (discarded) + N timed transcribes; keep the median latency and the
    transcript from a timed run. A failure here is recorded on the cell, not raised,
    so one bad clip doesn't abort the model or the run."""
    try:
        if warmup:
            backend.transcribe(clip.audio)  # discard: first call compiles CUDA kernels
        latencies: list[float] = []
        transcript = ""
        for _ in range(timed_runs):
            t0 = timer()
            transcript = backend.transcribe(clip.audio)  # capture from a *timed* run
            latencies.append(timer() - t0)
        median_latency = statistics.median(latencies)
        audio_s = len(clip.audio) / sample_rate
        rtf = median_latency / audio_s if audio_s > 0 else None
        return CellResult(
            clip_id=clip.id,
            transcript=transcript,
            score=score(clip.reference, transcript),
            median_latency_s=median_latency,
            rtf=rtf,
        )
    except Exception as exc:  # noqa: BLE001 (isolate one cell's failure)
        log(f"    {clip.id}: FAILED ({exc})")
        return CellResult(clip_id=clip.id, error=str(exc))


def _aggregate(spec: ModelSpec, load_s: float, cells: list[CellResult]) -> ModelResult:
    """Median of the successful cells per metric (None if every clip failed)."""
    ok = [c for c in cells if c.error is None]

    def med(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    return ModelResult(
        model_id=spec.id,
        label=spec.label,
        load_s=load_s,
        cells=cells,
        norm_wer=med([c.score.norm_wer for c in ok]),
        raw_wer=med([c.score.raw_wer for c in ok]),
        cer=med([c.score.cer for c in ok]),
        median_latency_s=med([c.median_latency_s for c in ok]),
        median_rtf=med([c.rtf for c in ok if c.rtf is not None]),
    )


def run_models(
    clips: list[Clip],
    specs: list[ModelSpec],
    *,
    timed_runs: int = 3,
    warmup: bool = True,
    timer: Callable[[], float] = time.perf_counter,
    sample_rate: int = SAMPLE_RATE,
    log: Callable[[str], None] = _default_log,
) -> RunResult:
    """Build each spec once (measuring load), then measure + score every clip.

    A spec whose `build()` raises (missing runtime/model) is skipped with its error
    recorded and omitted from the table; the remaining models still run."""
    models: list[ModelResult] = []
    for spec in specs:
        log(f"  building {spec.id} ...")
        t0 = timer()
        try:
            backend = spec.build()
        except Exception as exc:  # noqa: BLE001 (skip this model, keep going)
            log(f"  skipping {spec.id}: build failed ({exc})")
            models.append(ModelResult(model_id=spec.id, label=spec.label, build_error=str(exc)))
            continue
        load_s = timer() - t0
        cells = [
            _measure_clip(
                backend,
                clip,
                timed_runs=timed_runs,
                warmup=warmup,
                timer=timer,
                sample_rate=sample_rate,
                log=log,
            )
            for clip in clips
        ]
        models.append(_aggregate(spec, load_s, cells))
    return RunResult(clips=clips, models=models, sample_rate=sample_rate, timed_runs=timed_runs)


# --- report ----------------------------------------------------------------

_COLUMNS = "| model | norm WER | raw WER | CER | median latency (s) | median RTF | load (s) |"
_SEPARATOR = "|---|---|---|---|---|---|---|"


def _num(value: float | None, decimals: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def format_report(result: RunResult, *, timestamp: str | None = None) -> str:
    """Render the run as a Markdown report: a context header, one table row per
    model (build-failed models omitted, noted below), sorted in the order the specs
    were given (the baseline `fw-large-v3-turbo` first). Human-readable and diffable."""
    n = len(result.clips)
    total_audio_s = sum(len(c.audio) for c in result.clips) / result.sample_rate
    ran = [m for m in result.models if m.build_error is None]

    lines = [
        "# susurro-bench results",
        "",
        f"_{timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"{n} {'clip' if n == 1 else 'clips'}, {total_audio_s:.1f}s total audio · "
        f"aggregation: median across clips · "
        f"{result.timed_runs} timed runs per clip, warm-up discarded.",
        "",
        _COLUMNS,
        _SEPARATOR,
    ]
    for m in ran:
        lines.append(
            f"| {m.model_id} | {_num(m.norm_wer)} | {_num(m.raw_wer)} | {_num(m.cer)} "
            f"| {_num(m.median_latency_s)} | {_num(m.median_rtf)} | {_num(m.load_s, 1)} |"
        )
    lines.append("")

    notes = [f"- skipped **{m.model_id}**: {m.build_error}" for m in result.models if m.build_error]
    for m in ran:
        failed = [c.clip_id for c in m.cells if c.error is not None]
        if failed:
            notes.append(f"- **{m.model_id}** failed on clip(s): {', '.join(failed)}")
    lines.extend(notes)
    return "\n".join(lines).rstrip() + "\n"


# --- CLI body --------------------------------------------------------------


def run_command(args) -> int:
    """`susurro-bench run`: resolve selected+available models, discover the eval set,
    run, print the report to stdout, and write a timestamped copy to bench/results/."""
    from . import paths, registry
    from .cli import parse_models

    try:
        specs = registry.resolve(parse_models(args.models))
    except KeyError as exc:
        print(f"susurro-bench: {exc}", file=sys.stderr)
        return 2

    available = []
    for spec in specs:
        if spec.available():
            available.append(spec)
        else:
            print(f"susurro-bench: skipping {spec.id} (runtime not installed)", file=sys.stderr)
    if not available:
        print("susurro-bench: no selected models are available", file=sys.stderr)
        return 1

    clips, warnings = discover_clips(paths.scripts_dir(), paths.recordings_dir())
    for warning in warnings:
        print(f"susurro-bench: {warning}", file=sys.stderr)
    if not clips:
        print(
            "susurro-bench: no recordings found — run `susurro-bench record` first",
            file=sys.stderr,
        )
        return 1

    print(
        f"susurro-bench: {len(clips)} clip(s) x {len(available)} model(s); "
        "warm-up discarded, timing median of 3 ...",
        file=sys.stderr,
    )
    result = run_models(clips, available)
    report = format_report(result)
    print(report)

    results_dir = paths.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(report)
    print(f"susurro-bench: wrote {out}", file=sys.stderr)
    return 0
