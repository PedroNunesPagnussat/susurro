"""The UI-agnostic core: audio window -> clean text.

`Engine` owns the warm faster-whisper model (loaded once, reused) and exposes a
single `transcribe(audio) -> str`. Nothing UI- or driver-specific lives here, so
the daemon, the mic test, and the eval harness all share one transcription path.

This module also owns the one piece of *knowledge* about faster-whisper that
callers need before a model exists: which language codes it accepts
(`is_supported_language`). Whisper only validates the code deep inside
`transcribe`, which is far too late for a hold-to-talk UI — see that function.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from ._cuda import preload_cuda_libs
from .formatter import Formatter, RuleBasedFormatter

DEFAULT_MODEL = "large-v3-turbo"


@lru_cache(maxsize=1)
def _language_codes() -> frozenset[str] | None:
    """faster-whisper's static list of accepted language codes, or None if it can't
    be read.

    Deliberately defensive on two axes. (1) The import is lazy — like `Engine`'s —
    so `import susurro.engine` still doesn't drag in CTranslate2/CUDA for callers
    that only want the formatter or audio helpers. (2) `_LANGUAGE_CODES` is a
    *private* name (the only static list there is; `WhisperModel.supported_languages`
    needs a loaded model, which we may not have), so a future release can move or
    rename it. Returning None then degrades validation to "accept anything", which
    is exactly today's behaviour; a hard failure here would break dictation
    entirely, which is far worse than the mistake being guarded against.
    """
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
    except Exception:  # noqa: BLE001 (a moved/renamed private name must not break us)
        return None
    try:
        return frozenset(_LANGUAGE_CODES)
    except TypeError:  # not iterable any more -> same degrade-to-permissive path
        return None


def is_supported_language(code: str) -> bool:
    """True if faster-whisper will accept `code` as a transcription language.

    Whisper takes the language as a per-`transcribe` argument and only validates it
    when it builds the tokenizer, i.e. one utterance *after* the user asked for it
    (`faster_whisper/tokenizer.py` raises "'xx' is not a valid language code").
    Callers use this to reject a typo at the moment it's made, while the model is
    possibly not even loaded. Fails open: if the code list can't be read, every code
    is accepted and Whisper's own late error is the backstop (see `_language_codes`).
    Matching is exact, as Whisper's is — it does not lower-case the code either.
    """
    codes = _language_codes()
    return True if codes is None else code in codes


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
    once the last Python reference goes away). After an unload the daemon preloads
    on the key-press (`load()`), so the model build overlaps the hold; only the
    small first-inference CUDA warm lands on the release transcribe.
    """

    def __init__(
        self,
        factory: Callable[[], Engine],
        *,
        language: str = "en",
        log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr, flush=True),
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

    def load(self) -> None:
        """Build the inner engine now, without transcribing. No-op if loaded.

        The daemon calls this on the key-*press* so a post-idle model build
        overlaps the user speaking, instead of the whole cost landing on the
        release transcribe. The remembered language is applied to the fresh
        engine, exactly as a lazy (transcribe-triggered) build does."""
        if self._engine is None:
            self._log("susurro: loading model (cold start / post-idle) ...")
            self._engine = self._factory()
            # Apply the remembered language to the fresh engine: covers both the
            # first load (ctor arg) and every reload after an idle-unload.
            self._engine.set_language(self._language)

    def transcribe(self, audio: np.ndarray) -> str:
        self.load()  # cold-start / post-idle build; no-op when already warm
        return self._engine.transcribe(audio)

    def unload(self) -> None:
        """Drop the warm model and release its GPU memory. No-op if not loaded."""
        if self._engine is None:
            return
        self._engine = None
        gc.collect()  # run the CTranslate2 destructor now so VRAM frees promptly
        self._log("susurro: model unloaded (idle) — VRAM released")
