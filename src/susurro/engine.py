"""The UI-agnostic core: audio window -> clean text.

`Engine` owns the warm faster-whisper model (loaded once, reused) and exposes a
single `transcribe(audio) -> str`. Nothing UI- or driver-specific lives here, so
the daemon, the mic test, and the eval harness all share one transcription path.
"""

from __future__ import annotations

import gc
from collections.abc import Callable

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

    def set_language(self, code: str) -> None:
        """Switch the transcription language for subsequent `transcribe` calls.

        Language is a per-`transcribe` param, not baked into the loaded model, so
        this is a cheap live switch — no model reload."""
        self._language = code

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


class LazyEngine:
    """A reloadable wrapper around `Engine`: builds the model on first use and can
    drop it to release VRAM when idle, rebuilding lazily on the next `transcribe`.

    The daemon holds one of these so an idle daemon can free GPU memory. Dropping
    the inner `Engine` reference plus a `gc.collect()` releases the CTranslate2
    model's CUDA allocation (CTranslate2 frees it in the C++ destructor, which runs
    once the last Python reference goes away). The first utterance after an unload
    pays the full model-load + CUDA-warm cost — the accepted tradeoff.
    """

    def __init__(
        self,
        factory: Callable[[], Engine],
        *,
        language: str = "en",
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ) -> None:
        self._factory = factory
        self._language = language
        self._log = log
        self._engine: Engine | None = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def set_language(self, code: str) -> None:
        """Remember the language and apply it to the engine if one is loaded.

        The stored code is re-applied whenever the engine is (re)built, so an
        idle-unload->reload keeps the chosen language instead of reverting to the
        factory default."""
        self._language = code
        if self._engine is not None:
            self._engine.set_language(code)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._engine is None:
            self._log("susurro: loading model (cold start / post-idle) ...")
            self._engine = self._factory()
            # Apply the remembered language to the fresh engine: covers both the
            # first load (ctor arg) and every reload after an idle-unload.
            self._engine.set_language(self._language)
        return self._engine.transcribe(audio)

    def unload(self) -> None:
        """Drop the warm model and release its GPU memory. No-op if not loaded."""
        if self._engine is None:
            return
        self._engine = None
        gc.collect()  # run the CTranslate2 destructor now so VRAM frees promptly
        self._log("susurro: model unloaded (idle) — VRAM released")
