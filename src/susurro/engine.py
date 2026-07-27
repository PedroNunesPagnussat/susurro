"""The UI-agnostic core: audio window -> clean text.

`Engine` owns the warm faster-whisper model (loaded once, reused) and exposes a
single `transcribe(audio) -> str`. Nothing UI- or driver-specific lives here, so
the daemon, the mic test, and the eval harness all share one transcription path.

It also owns the two things callers need before a model exists: which language codes
faster-whisper accepts (`is_supported_language`), and how an `[engine]` config
becomes a warm engine (`build_engine` / `load_engine`). Both entrypoints go through
those, so adding an engine knob is one edit here, not a change to the CLI module.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ._cuda import preload_cuda_libs
from .formatter import Formatter, RuleBasedFormatter

if TYPE_CHECKING:
    # Type-only: `config` imports `DEFAULT_MODEL` from here, so a real import would
    # be a cycle. `from __future__ import annotations` keeps the annotation a string.
    from .config import EngineConfig

DEFAULT_MODEL = "large-v3-turbo"


@lru_cache(maxsize=1)
def _language_codes() -> frozenset[str] | None:
    """faster-whisper's static list of accepted language codes, or None if it can't
    be read.

    Lazy import — like `Engine`'s — so `import susurro.engine` doesn't drag in
    CTranslate2/CUDA for callers that only want the formatter or audio helpers.
    `_LANGUAGE_CODES` is *private* (the only static list there is;
    `WhisperModel.supported_languages` needs a loaded model), so a future release
    can move it. Returning None degrades validation to "accept anything" rather
    than breaking dictation entirely.
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

    Whisper only validates the language when it builds the tokenizer, i.e. one
    utterance *after* the user asked for it. Callers use this to reject a typo at the
    moment it's made, while the model may not even be loaded. Fails open: if the code
    list can't be read, every code is accepted and Whisper's own late error is the
    backstop. Matching is exact, as Whisper's is.
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


class EngineLoadError(RuntimeError):
    """Startup failure: the model couldn't be built or warmed (bad model name, failed
    download, broken CUDA install). Its message already names the model and the device
    — the two knobs — so a caller can print it without re-deriving the context."""


def build_engine(cfg: EngineConfig, *, lazy: bool = False) -> Engine | LazyEngine:
    """An engine from the resolved `[engine]` config — the one place those fields map
    onto the constructor, so a new knob is a single edit.

    `lazy=True` wraps it in a `LazyEngine` (what the daemon holds, so an idle daemon
    can drop the model and free VRAM); the mic test wants the plain `Engine`. Both
    shapes live here because `LazyEngine` does.
    """

    def factory() -> Engine:
        return Engine(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
        )

    return LazyEngine(factory, language=cfg.language) if lazy else factory()


def load_engine(cfg: EngineConfig, *, sample_rate: int, lazy: bool = False) -> Engine | LazyEngine:
    """Build the engine and warm it, or raise `EngineLoadError`.

    The warmup transcribe is where a bad model name, a failed download or a broken
    CUDA install surfaces — a `LazyEngine` defers the build to its first transcribe,
    so without it the daemon would report the failure one utterance later — and it
    leaves the kernels compiled, so the first real utterance already hits warm timing.
    """
    try:
        engine = build_engine(cfg, lazy=lazy)
        engine.transcribe(np.zeros(sample_rate // 2, dtype=np.float32))
    except Exception as exc:
        raise EngineLoadError(f"failed to load model {cfg.model!r} on {cfg.device}: {exc}") from exc
    return engine
