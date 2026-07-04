"""Make the venv-local NVIDIA CUDA wheels loadable by CTranslate2.

The ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels drop their ``.so`` files
under ``site-packages/nvidia/*/lib`` but do not register them with the dynamic
loader. CTranslate2 ``dlopen``s libcublas and libcudnn by soname, so we preload
them here (``RTLD_GLOBAL``) before the engine constructs a CUDA model.

Setting ``LD_LIBRARY_PATH`` from inside the process is unreliable — glibc parses
it once at startup — so we preload explicitly with absolute paths instead. This
keeps the venv self-contained (no external env wiring), matching the plan's
decision that "the venv owns its CUDA stack."
"""

from __future__ import annotations

import ctypes
import glob
import os

# cuBLAS is listed first because cuDNN links against it; the actual load order is
# resolved by retry-until-stable below, so this is only a hint.
_PACKAGES = ("cublas", "cudnn")


def _nvidia_lib_dirs() -> list[str]:
    try:
        import nvidia
    except ImportError:
        return []
    # `nvidia` is a namespace package (no __init__), so use __path__, not __file__.
    base = next(iter(nvidia.__path__), None)
    if base is None:
        return []
    return [os.path.join(base, pkg, "lib") for pkg in _PACKAGES]


def preload_cuda_libs() -> list[str]:
    """Preload the venv's cuBLAS/cuDNN shared objects into the global symbol scope.

    Returns the list of loaded library paths (empty if the wheels aren't present,
    e.g. on a CPU-only install). Failures are swallowed so CTranslate2 can surface
    the real, more informative CUDA error itself.
    """
    candidates: list[str] = []
    for libdir in _nvidia_lib_dirs():
        candidates.extend(sorted(glob.glob(os.path.join(libdir, "*.so*"))))

    loaded: list[str] = []
    pending = list(candidates)
    # Retry until no further progress: this resolves inter-library load order
    # (e.g. a cuDNN component needing cuBLAS symbols) without hardcoding a graph.
    while pending:
        still: list[str] = []
        progressed = False
        for so in pending:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(so)
                progressed = True
            except OSError:
                still.append(so)
        pending = still
        if not progressed:
            break
    return loaded
