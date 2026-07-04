"""Engine transcribe-seam test.

Runs the real model against the committed JFK fixture and asserts the expected
words appear. Skipped when no CUDA device is present (CI / other machines) so it
stays a local confidence check, not a hard gate — per the plan's Testing Decisions.
"""

from pathlib import Path

import pytest

from susurro.audio import load_wav

FIXTURE = Path(__file__).parent / "fixtures" / "jfk_16k_mono.wav"


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="no CUDA device; engine seam test is a local check"
)


@pytest.fixture(scope="module")
def engine():
    from susurro.engine import Engine

    return Engine()  # warm model, loaded once for the module


def test_transcribes_known_speech(engine):
    audio = load_wav(FIXTURE)
    text = engine.transcribe(audio).lower()
    # The JFK clip: "...ask not what your country can do for you..."
    assert "country" in text
    assert "americans" in text


def test_silence_yields_empty(engine):
    import numpy as np

    silence = np.zeros(16_000 * 2, dtype=np.float32)  # 2s of digital silence
    assert engine.transcribe(silence) == ""
