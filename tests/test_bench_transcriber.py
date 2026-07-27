"""The `Transcriber` protocol is the runner's only contract with a backend, so
lock its structural shape: `transcribe` alone makes you a Transcriber (the report
names models from the registry, so the protocol carries nothing else), missing
transcribe doesn't. Prior art: `_ManagedEngine` (`src/susurro/daemon.py`), the
other protocol in this repo that is isinstance-checked rather than static-only.
"""

import numpy as np

from susurro.bench.transcriber import Transcriber


class _Canned:
    """A minimal backend: just a transcribe. This is exactly what the runner's
    FakeTranscriber (Step 5) and the real adapters provide."""

    def transcribe(self, audio: np.ndarray) -> str:
        return "hello"


class _NoTranscribe:
    """Everything a backend might carry *except* the one required method."""

    name = "broken"


def test_object_with_only_transcribe_is_a_transcriber():
    assert isinstance(_Canned(), Transcriber)


def test_object_missing_transcribe_is_not_a_transcriber():
    # Not even with other attributes: `transcribe` is the whole contract.
    assert not isinstance(_NoTranscribe(), Transcriber)
