"""The common seam every benchmark backend implements.

`Transcriber` is a Protocol like `Formatter` (`src/susurro/formatter.py`): a
structural type so the runner stays backend-blind (no per-backend if/elif). Every
backend — faster-whisper, whisper.cpp — satisfies it by having a `transcribe`, and
nothing else. It is `runtime_checkable` (like `_ManagedEngine` in
`src/susurro/daemon.py`, unlike `Formatter`) because that shape is pinned by an
`isinstance` test in `tests/test_bench_transcriber.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Transcriber(Protocol):
    """A warm model that turns one audio window into text.

    `transcribe` takes **16kHz mono float32** — the format the harness guarantees
    for every clip (from `Recorder`/`load_wav`), so no backend resamples. The
    report names each model by its registry `ModelSpec.id`, not by the backend
    object, so the protocol is just this one method.
    """

    def transcribe(self, audio: np.ndarray) -> str: ...
