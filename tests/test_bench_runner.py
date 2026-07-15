"""Unit tests for the benchmark runner — the orchestration, exercised with a
`FakeTranscriber` + a fake clock so no CUDA and no mic are touched. Prior art:
`FakeManagedEngine`/`FakeClock` (`tests/test_daemon.py`), fake factory
(`tests/test_lazy_engine.py`).

Skipped without the `bench` extra (the runner scores via jiwer).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jiwer")

from susurro.audio import SAMPLE_RATE, load_wav  # noqa: E402
from susurro.bench.recording import save_wav  # noqa: E402
from susurro.bench.registry import ModelSpec  # noqa: E402
from susurro.bench.runner import (  # noqa: E402
    Clip,
    discover_clips,
    format_report,
    run_models,
)

FIXTURE = Path(__file__).parent / "fixtures" / "jfk_16k_mono.wav"


class FakeClock:
    """Monotonic fake time; the FakeTranscriber advances it so timed deltas are
    deterministic without real sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeTranscriber:
    """Canned transcript + scripted per-call durations. Each `transcribe` advances
    the shared clock by the next duration, so the runner's perf_counter deltas are
    exactly those durations (warm-up first, then the N timed runs)."""

    def __init__(self, name, clock, durations, transcript="hello world", fail=False):
        self.name = name
        self._clock = clock
        self._durations = list(durations)
        self._transcript = transcript
        self._fail = fail
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        self._clock.advance(self._durations[self.calls - 1])
        if self._fail:
            raise RuntimeError("boom")
        return self._transcript


def _spec(transcriber, *, id="fw-fake", label="fake", load_dt=0.0, clock=None, fail_build=False):
    """A ModelSpec whose build() returns `transcriber` (optionally advancing the
    clock by load_dt to simulate model load, or raising to simulate a build fail)."""

    def build():
        if fail_build:
            raise RuntimeError("no runtime")
        if clock is not None:
            clock.advance(load_dt)
        return transcriber

    return ModelSpec(id=id, label=label, build=build, available=lambda: True)


# --- discovery -------------------------------------------------------------


def test_discover_pairs_scripts_and_recordings_by_stem(tmp_path):
    scripts = tmp_path / "scripts"
    recs = tmp_path / "recordings"
    scripts.mkdir()
    recs.mkdir()
    (scripts / "a.txt").write_text("alpha reference")
    (scripts / "b.txt").write_text("bravo reference")
    (scripts / "c.txt").write_text("charlie has no recording")  # missing recording
    save_wav(recs / "a.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))
    save_wav(recs / "b.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))
    save_wav(recs / "d.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))  # orphan recording

    clips, warnings = discover_clips(scripts, recs)

    assert [c.id for c in clips] == ["a", "b"]  # paired, sorted
    assert clips[0].reference == "alpha reference"
    assert clips[0].audio.shape == (SAMPLE_RATE,)
    warn_text = " ".join(warnings)
    assert "c" in warn_text  # script with no recording is flagged
    assert "d" in warn_text  # orphan recording is flagged


# --- measurement: warm-up + median-of-N + RTF ------------------------------


def test_discards_warmup_and_keeps_median_of_n_timed_runs():
    clock = FakeClock()
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # exactly 2.0s
    clip = Clip(id="clip", reference="hello world", audio=audio)
    # warm-up 9.0 (discarded), then timed 0.2 / 0.5 / 0.3 -> median 0.3.
    fake = FakeTranscriber("fw-fake", clock, durations=[9.0, 0.2, 0.5, 0.3])

    result = run_models([clip], [_spec(fake)], timed_runs=3, timer=clock)

    assert fake.calls == 4  # 1 warm-up + 3 timed, not a 4th untimed capture
    cell = result.models[0].cells[0]
    assert cell.median_latency_s == pytest.approx(0.3)
    assert cell.rtf == pytest.approx(0.3 / 2.0)  # median latency / audio seconds


def test_scores_the_captured_transcript_with_the_real_scorer():
    clock = FakeClock()
    clip = Clip(id="clip", reference="the quick brown fox", audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
    fake = FakeTranscriber("fw-fake", clock, durations=[1.0, 0.1, 0.1, 0.1], transcript="the quick brown cat")

    result = run_models([clip], [_spec(fake)], timed_runs=3, timer=clock)

    cell = result.models[0].cells[0]
    assert cell.transcript == "the quick brown cat"
    assert cell.score.norm_wer == pytest.approx(0.25)  # one word wrong in four


def test_load_time_is_measured_separately_from_latency():
    clock = FakeClock()
    clip = Clip(id="clip", reference="hi", audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
    fake = FakeTranscriber("fw-fake", clock, durations=[1.0, 0.1, 0.1, 0.1])

    result = run_models([clip], [_spec(fake, load_dt=2.5, clock=clock)], timed_runs=3, timer=clock)

    assert result.models[0].load_s == pytest.approx(2.5)


def test_aggregates_wer_as_median_across_clips():
    clock = FakeClock()
    # clip1 perfect (wer 0.0), clip2 one wrong word of two (wer 0.5) -> median 0.25.
    clips = [
        Clip("c1", "hello world", np.zeros(SAMPLE_RATE, dtype=np.float32)),
        Clip("c2", "hello world", np.zeros(SAMPLE_RATE, dtype=np.float32)),
    ]

    class TwoClipFake(FakeTranscriber):
        def transcribe(self, audio):
            self.calls += 1
            self._clock.advance(0.1)
            # first clip's 4 calls -> perfect; second clip's -> one wrong word.
            return "hello world" if self.calls <= 4 else "hello mars"

    fake = TwoClipFake("fw-fake", clock, durations=[0.1] * 8)
    result = run_models(clips, [_spec(fake)], timed_runs=3, timer=clock)

    assert result.models[0].norm_wer == pytest.approx(0.25)


# --- resilience ------------------------------------------------------------


def test_build_failure_skips_model_and_the_rest_still_report():
    clock = FakeClock()
    clip = Clip("c", "hello world", np.zeros(SAMPLE_RATE, dtype=np.float32))
    good = FakeTranscriber("fw-good", clock, durations=[1.0, 0.1, 0.1, 0.1])
    specs = [
        _spec(good, id="fw-broken", label="broken", fail_build=True),
        _spec(good, id="fw-good", label="good"),
    ]

    result = run_models([clip], specs, timed_runs=3, timer=clock)

    broken, ok = result.models
    assert broken.build_error is not None  # recorded, not raised
    assert broken.norm_wer is None
    assert ok.build_error is None and ok.norm_wer is not None  # the run went on
    # a build-failed model is omitted from the table but noted below it.
    report = format_report(result)
    assert "fw-good" in report
    assert "fw-broken" not in _table_body(report)


def test_single_clip_error_is_isolated_not_a_whole_run_abort():
    clock = FakeClock()
    clips = [
        Clip("ok", "hello world", np.zeros(SAMPLE_RATE, dtype=np.float32)),
        Clip("bad", "hello world", np.zeros(SAMPLE_RATE, dtype=np.float32)),
    ]

    class FailSecondClip(FakeTranscriber):
        def transcribe(self, audio):
            self.calls += 1
            self._clock.advance(0.1)
            if self.calls > 4:  # everything after the first clip's 4 calls
                raise RuntimeError("clip blew up")
            return "hello world"

    fake = FailSecondClip("fw-fake", clock, durations=[0.1] * 8)
    result = run_models(clips, [_spec(fake)], timed_runs=3, timer=clock)

    cells = {c.clip_id: c for c in result.models[0].cells}
    assert cells["ok"].error is None
    assert cells["bad"].error is not None
    assert result.models[0].norm_wer is not None  # aggregated over the good clip only


# --- report ----------------------------------------------------------------


def _table_body(report: str) -> str:
    """Just the Markdown table rows (lines starting with '|')."""
    return "\n".join(line for line in report.splitlines() if line.startswith("|"))


def test_report_keeps_spec_order_and_headers_the_clip_count():
    clock = FakeClock()
    clip = Clip("c", "hello world", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    a = FakeTranscriber("fw-a", clock, durations=[1.0, 0.1, 0.1, 0.1])
    b = FakeTranscriber("fw-b", clock, durations=[1.0, 0.1, 0.1, 0.1])
    specs = [_spec(a, id="fw-baseline", label="baseline"), _spec(b, id="fw-other", label="other")]

    result = run_models([clip], specs, timed_runs=3, timer=clock)
    report = format_report(result)

    body = _table_body(report)
    assert body.index("fw-baseline") < body.index("fw-other")  # baseline (spec order) first
    assert "1 clip" in report  # header states clip count
    assert "median" in report.lower()  # aggregation is stated


def test_report_over_the_committed_jfk_fixture_produces_a_full_row():
    # The step names this fixture: drive the runner over it (fake model, real audio)
    # and confirm a full table row renders end-to-end without CUDA or a mic.
    clock = FakeClock()
    audio = load_wav(FIXTURE)
    clip = Clip("jfk", "and so my fellow americans", audio)
    fake = FakeTranscriber("fw-fake", clock, durations=[9.0, 0.4, 0.4, 0.4], transcript="and so my fellow americans")

    result = run_models([clip], [_spec(fake, id="fw-large-v3-turbo", label="turbo")], timed_runs=3, timer=clock)
    report = format_report(result)

    assert "fw-large-v3-turbo" in report
    assert "0.0" in report  # a perfect transcript -> 0.0 WER shows in the row
