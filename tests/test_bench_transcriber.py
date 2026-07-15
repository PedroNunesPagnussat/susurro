"""The `Transcriber` protocol is the runner's only contract with a backend, so
lock its structural shape: name + transcribe makes you a Transcriber, missing
transcribe doesn't. Prior art: the runtime_checkable `Formatter`
(`src/susurro/formatter.py`).
"""

import numpy as np

from susurro.bench.transcriber import Transcriber


class _Canned:
    """A minimal backend: a name and a transcribe. This is exactly what the
    runner's FakeTranscriber (Step 5) and the real adapters provide."""

    name = "canned"

    def transcribe(self, audio: np.ndarray) -> str:
        return "hello"


class _NoTranscribe:
    name = "broken"


def test_object_with_name_and_transcribe_is_a_transcriber():
    assert isinstance(_Canned(), Transcriber)


def test_object_missing_transcribe_is_not_a_transcriber():
    assert not isinstance(_NoTranscribe(), Transcriber)
