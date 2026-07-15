"""whisper.cpp backend (optional): `large-v3-turbo` GGUF via `pywhispercpp`.

Lazy-import by contract — `pywhispercpp` is a `bench-whispercpp` extra, not a core
dep — so importing this module without the runtime installed raises `ImportError`,
which the registry turns into "unavailable" (an import probe) and the runner turns
into a clean skip. The model is built warm in `__init__` like the other backends.

whisper.cpp consumes 16kHz mono float32 directly (the harness format), so there's
no resampling. Verified by hand when installed; not unit-tested (real runtime).
"""

from __future__ import annotations

import numpy as np

from ..formatter import RuleBasedFormatter


class WhisperCppTranscriber:
    """A `Transcriber` backed by a warm whisper.cpp GGUF model (via pywhispercpp)."""

    def __init__(self, model_name: str = "large-v3-turbo", *, language: str = "en") -> None:
        from pywhispercpp.model import Model  # lazy: optional runtime

        self.name = f"whispercpp-{model_name}"
        self._formatter = RuleBasedFormatter()
        # pywhispercpp resolves the GGUF by shorthand name (downloading once) and
        # keeps it resident; the quiet flags stop it printing per-segment progress.
        self._model = Model(
            model_name,
            language=language,
            print_progress=False,
            print_realtime=False,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments = self._model.transcribe(audio)
        raw = " ".join(segment.text for segment in segments)
        # Same whitespace cleanup the fw path applies via Engine, so every backend is
        # scored on comparably-formatted "what gets typed" text.
        return self._formatter.format(raw)
