"""Where the benchmark's files live, resolved relative to this source so the paths
hold regardless of the working directory the CLI was launched from (mirrors
`config.default_config_path`).

    bench/scripts/     committed reference text  (<id>.txt)
    bench/recordings/  the owner's voice takes   (<id>.wav, gitignored)
    bench/results/     regenerated report files  (gitignored)
"""

from __future__ import annotations

from pathlib import Path

# src/susurro/bench/paths.py -> parents[3] is the repo root. Module-level so a test
# can point it elsewhere and prove the guard below fires.
_SOURCE = Path(__file__).resolve()


class BenchPathError(RuntimeError):
    """The bench tree isn't where it should be. Its own type so `cli.main` can turn
    it into a one-line message without also swallowing an unrelated `RuntimeError`
    (`Recorder.start` raises one for a busy/failed capture device)."""


def bench_dir() -> Path:
    """The repo's `bench/` directory. Requires the source checkout (`uv sync`
    installs the package editable, so `__file__` stays in the repo); a non-editable
    install would resolve somewhere under site-packages, so check rather than let
    `results_dir().mkdir(parents=True)` scatter directories outside the repo."""
    d = _SOURCE.parents[3] / "bench"
    if not d.is_dir():
        raise BenchPathError(f"bench directory not found at {d}: susurro-bench needs the repo")
    return d


def scripts_dir() -> Path:
    return bench_dir() / "scripts"


def recordings_dir() -> Path:
    return bench_dir() / "recordings"


def results_dir() -> Path:
    return bench_dir() / "results"
