"""The bench paths are resolved from this source file's location, which only holds
in the source checkout (`uv sync` installs the package editable). These tests pin
both halves: the happy path points at the repo's own `bench/`, and a resolution that
lands anywhere else fails loud instead of silently writing outside the repo.
"""

import pytest

from susurro.bench import paths


def test_bench_dirs_resolve_to_the_repo_bench_tree():
    bench = paths.bench_dir()
    assert bench.name == "bench"
    assert (bench / "scripts").is_dir()  # the committed reference text
    assert paths.scripts_dir() == bench / "scripts"
    assert paths.recordings_dir() == bench / "recordings"
    assert paths.results_dir() == bench / "results"


def test_bench_dir_raises_when_resolution_lands_outside_the_repo(monkeypatch, tmp_path):
    # What a non-editable install looks like: parents[3] resolves into site-packages,
    # where there is no bench/ — and `run` would otherwise mkdir a results tree there.
    monkeypatch.setattr(paths, "_SOURCE", tmp_path / "a" / "b" / "c" / "paths.py")
    with pytest.raises(paths.BenchPathError, match="bench directory not found"):
        paths.bench_dir()


def test_cli_reports_a_missing_bench_tree_without_a_traceback(monkeypatch, tmp_path, capsys):
    # The guard fires deep inside `run`; the user should see the house one-liner, not
    # a stack trace. Anything else raising RuntimeError must still propagate.
    from susurro.bench import cli

    monkeypatch.setattr(paths, "_SOURCE", tmp_path / "a" / "b" / "c" / "paths.py")
    assert cli.main(["run"]) == 1
    err = capsys.readouterr().err
    assert "susurro-bench: bench directory not found" in err
    assert "Traceback" not in err
