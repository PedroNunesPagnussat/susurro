"""Unit tests for the CUDA preloader (`susurro._cuda`).

No real CUDA wheels are dlopen'd: `ctypes.CDLL`, `glob.glob`, and the nvidia lib
dirs are all faked. The behaviour under test is the retry-until-stable loader —
it must resolve inter-library load order, stop when it stalls, and no-op cleanly
when the wheels aren't installed (CPU-only box).
"""

import sys

from susurro import _cuda


def _patch(monkeypatch, *, libdirs, glob_returns, cdll):
    monkeypatch.setattr(_cuda, "_nvidia_lib_dirs", lambda: libdirs)
    monkeypatch.setattr(_cuda.glob, "glob", lambda pattern: list(glob_returns))
    monkeypatch.setattr(_cuda.ctypes, "CDLL", cdll)


def test_returns_empty_when_no_lib_dirs(monkeypatch):
    # CPU-only install: the nvidia wheels aren't present, so there's nothing to load.
    _patch(monkeypatch, libdirs=[], glob_returns=[], cdll=lambda *a, **k: None)
    assert _cuda.preload_cuda_libs() == []


def test_returns_empty_when_dir_has_no_libs(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("CDLL must not be called when there are no candidates")

    _patch(monkeypatch, libdirs=["/fake/lib"], glob_returns=[], cdll=boom)
    assert _cuda.preload_cuda_libs() == []


def test_retries_until_dependency_order_resolves(monkeypatch):
    # A.so needs B.so loaded first, but sorts *before* it — so the first pass fails
    # on A, loads B, and the retry then loads A. Proves the loop doesn't depend on
    # candidate order.
    loaded = set()

    def cdll(path, mode=0):
        if path.endswith("A.so") and "/fake/B.so" not in loaded:
            raise OSError("A needs B loaded first")
        loaded.add(path)

    _patch(monkeypatch, libdirs=["/fake"], glob_returns=["/fake/A.so", "/fake/B.so"], cdll=cdll)
    result = _cuda.preload_cuda_libs()
    assert set(result) == {"/fake/A.so", "/fake/B.so"}
    assert result.index("/fake/B.so") < result.index("/fake/A.so")  # B first, A on retry


def test_stops_when_no_further_progress(monkeypatch):
    # A permanently-unloadable lib must not loop forever: once a pass makes no
    # progress, the loader gives up and returns what it managed to load.
    def cdll(path, mode=0):
        if "bad" in path:
            raise OSError("never loads")

    _patch(
        monkeypatch, libdirs=["/fake"], glob_returns=["/fake/bad.so", "/fake/good.so"], cdll=cdll
    )
    assert _cuda.preload_cuda_libs() == ["/fake/good.so"]


def test_nvidia_lib_dirs_empty_when_package_missing(monkeypatch):
    # `import nvidia` failing (None in sys.modules raises ImportError) must yield an
    # empty dir list, not blow up — the CPU-only path.
    monkeypatch.setitem(sys.modules, "nvidia", None)
    assert _cuda._nvidia_lib_dirs() == []
