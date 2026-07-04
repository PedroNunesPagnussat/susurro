"""The UI-agnostic core: audio window -> clean text.

`Engine` owns the warm faster-whisper model (loaded once, reused) and runs the
agreed Phase-1 defaults. Everything UI/driver-specific stays out of here so Phase 2
can reuse the Engine unchanged behind a hotkey + Unix socket.
"""

from __future__ import annotations

import numpy as np

from ._cuda import preload_cuda_libs
from .formatter import Formatter, RuleBasedFormatter

DEFAULT_MODEL = "large-v3-turbo"


class Engine:
    """Warm Whisper model + formatter. Call `transcribe(audio)` per utterance."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
        formatter: Formatter | None = None,
    ) -> None:
        # Import here so `import susurro.engine` doesn't drag in CTranslate2/CUDA
        # for callers that only touch the formatter or audio helpers.
        from faster_whisper import WhisperModel

        if device == "cuda":
            preload_cuda_libs()

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._formatter: Formatter = formatter or RuleBasedFormatter()
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one mono float32 window and return formatted text ("" if none)."""
        if audio is None or len(audio) == 0:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            # Independent utterances: no context bleed / repetition across windows.
            condition_on_previous_text=False,
        )
        raw = " ".join(segment.text for segment in segments)
        return self._formatter.format(raw)
