"""The model registry: an ordered `id -> ModelSpec` map the runner and
`--list-models` read.

A `ModelSpec` carries a human label, a lazy `build()` factory (returns a
`Transcriber`), and an `available()` predicate. Availability is a pure *import
probe* (`importlib.util.find_spec`) — it never constructs a model, downloads
weights, or touches CUDA — so `--list-models` and the runner's skip logic stay
cheap and side-effect-free. Building a real backend happens only inside `build()`,
i.e. only when a run actually selects that model.

Ordered so the baseline (`fw-large-v3-turbo`) is first, which the report relies on.
The two faster-whisper ids are registered now; whisper.cpp (Step 6) and Parakeet
(Step 7) slot in below when their backends land.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

from .transcriber import Transcriber


@dataclass(frozen=True)
class ModelSpec:
    """One benchmarkable model: how to name it, build it, and whether it can run."""

    id: str
    label: str
    build: Callable[[], Transcriber]
    available: Callable[[], bool]


def _importable(module: str) -> bool:
    """True if `module` can be imported, without importing it. The availability
    probe for every backend — cheap and side-effect-free (no CUDA, no download)."""
    return importlib.util.find_spec(module) is not None


def _build_faster_whisper(model_name: str) -> Callable[[], Transcriber]:
    """A lazy factory for a faster-whisper contender: imports the backend (and
    thus CTranslate2/CUDA) only when called, never at registry import."""

    def build() -> Transcriber:
        from .faster_whisper_backend import FasterWhisperTranscriber

        return FasterWhisperTranscriber(model_name)

    return build


def _faster_whisper_available() -> bool:
    return _importable("faster_whisper")


def _build_whispercpp(model_name: str) -> Callable[[], Transcriber]:
    """Lazy factory for the whisper.cpp backend: imports pywhispercpp (the optional
    runtime) only when called, so an uninstalled runtime never blocks registry import."""

    def build() -> Transcriber:
        from .whispercpp_backend import WhisperCppTranscriber

        return WhisperCppTranscriber(model_name)

    return build


def _build_parakeet(model_name: str) -> Callable[[], Transcriber]:
    """Lazy factory for the Parakeet backend: imports NeMo (the optional runtime,
    which pulls torch) only when called, never at registry import."""

    def build() -> Transcriber:
        from .parakeet_backend import ParakeetTranscriber

        return ParakeetTranscriber(model_name)

    return build


# Insertion order is the report order: baseline turbo first, then large-v3.
# whisper.cpp / Parakeet register here in Steps 6-7 with their own lazy factory +
# import-probe availability, so an uninstalled runtime simply reports unavailable.
REGISTRY: dict[str, ModelSpec] = {
    "fw-large-v3-turbo": ModelSpec(
        id="fw-large-v3-turbo",
        label="faster-whisper large-v3-turbo (int8/CUDA)",
        build=_build_faster_whisper("large-v3-turbo"),
        available=_faster_whisper_available,
    ),
    "fw-large-v3": ModelSpec(
        id="fw-large-v3",
        label="faster-whisper large-v3 (int8/CUDA)",
        build=_build_faster_whisper("large-v3"),
        available=_faster_whisper_available,
    ),
    "whispercpp-turbo": ModelSpec(
        id="whispercpp-turbo",
        label="whisper.cpp large-v3-turbo (GGUF)",
        build=_build_whispercpp("large-v3-turbo"),
        available=lambda: _importable("pywhispercpp"),
    ),
    "parakeet-tdt-0.6b-v2": ModelSpec(
        id="parakeet-tdt-0.6b-v2",
        label="NVIDIA Parakeet-TDT 0.6B v2 (NeMo, English-only)",
        build=_build_parakeet("nvidia/parakeet-tdt-0.6b-v2"),
        available=lambda: _importable("nemo"),
    ),
}


def resolve(ids: list[str] | None) -> list[ModelSpec]:
    """Specs for `ids` (or every registered model when None), in registry order.

    An unknown id raises `KeyError` naming it and listing the valid ids, so a
    typo'd `--models` fails loudly at startup instead of silently benchmarking
    fewer models than asked."""
    if ids is None:
        return list(REGISTRY.values())
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        known = ", ".join(REGISTRY)
        raise KeyError(f"unknown model id(s) {unknown}; known ids: {known}")
    # Registry order, not argument order, so the report is deterministic.
    return [spec for spec in REGISTRY.values() if spec.id in set(ids)]
