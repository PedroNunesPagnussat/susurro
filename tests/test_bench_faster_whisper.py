"""Real-model seam test for the faster-whisper bench adapter.

Skipped without CUDA (CI / other machines), exactly like `tests/test_engine.py`:
it's a local confidence check that the adapter builds a real model, satisfies the
`Transcriber` protocol, and delegates transcription — not a hard gate.
"""

from pathlib import Path

import pytest

from susurro.audio import load_wav
from susurro.bench.transcriber import Transcriber

FIXTURE = Path(__file__).parent / "fixtures" / "jfk_16k_mono.wav"


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="no CUDA device; fw adapter seam test is a local check"
)


def test_adapter_is_a_transcriber_and_transcribes_known_speech():
    from susurro.bench.faster_whisper_backend import FasterWhisperTranscriber

    backend = FasterWhisperTranscriber("large-v3-turbo")
    assert isinstance(backend, Transcriber)  # satisfies the runner's contract
    assert backend.name == "fw-large-v3-turbo"

    text = backend.transcribe(load_wav(FIXTURE)).lower()
    assert "country" in text
    assert "americans" in text
