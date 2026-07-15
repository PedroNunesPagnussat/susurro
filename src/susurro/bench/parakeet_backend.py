"""Parakeet backend (optional): NVIDIA `parakeet-tdt-0.6b-v2` via NeMo.

Lazy-import by contract — the NeMo stack (`nemo-toolkit[asr]`, which pulls torch)
is a `bench-parakeet` extra, not a core dep — so importing this module without NeMo
installed raises `ImportError`, which the registry reads as "unavailable" and the
runner turns into a clean skip. The model is built warm in `__init__` like the
other backends and moved to CUDA.

Parakeet-v2 is English-only, which is why the whole benchmark is English-only (spec
Decisions). It consumes 16kHz mono float32 (the harness format), no resampling.
Verified by hand when installed; not unit-tested (heavy real runtime).
"""

from __future__ import annotations

import numpy as np

from ..formatter import RuleBasedFormatter


def _first_text(outputs: object) -> str:
    """Pull the transcript out of NeMo's `transcribe` return, which varies by
    version: a list[str], a list[Hypothesis] (each with `.text`), or a tuple whose
    first element is one of those (some RNNT/TDT paths)."""
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    item = outputs[0]  # batch of one
    return item.text if hasattr(item, "text") else str(item)


class ParakeetTranscriber:
    """A `Transcriber` backed by a warm NeMo Parakeet-TDT model on CUDA."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        *,
        device: str = "cuda",
    ) -> None:
        import nemo.collections.asr as nemo_asr  # lazy: optional runtime (pulls torch)

        self.name = "parakeet-tdt-0.6b-v2"
        self._formatter = RuleBasedFormatter()
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        # Wrap the single clip in a batch of one; NeMo accepts 16kHz mono float32.
        outputs = self._model.transcribe([audio], batch_size=1)
        return self._formatter.format(_first_text(outputs))
