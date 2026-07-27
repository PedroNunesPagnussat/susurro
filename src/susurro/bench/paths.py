"""Where the benchmark's files live, resolved relative to this source so the paths
hold regardless of the working directory the CLI was launched from (mirrors
`config.default_config_path`).

    bench/scripts/     committed reference text  (<id>.txt)
    bench/recordings/  the owner's voice takes   (<id>.wav, gitignored)
    bench/results/     regenerated report files  (gitignored)
"""

from __future__ import annotations

from pathlib import Path


def bench_dir() -> Path:
    # src/susurro/bench/paths.py -> parents[3] is the repo root.
    return Path(__file__).resolve().parents[3] / "bench"


def scripts_dir() -> Path:
    return bench_dir() / "scripts"


def recordings_dir() -> Path:
    return bench_dir() / "recordings"


def results_dir() -> Path:
    return bench_dir() / "results"
