"""faster-whisper backend: the two contenders that run out of the box.

Both `large-v3-turbo` and `large-v3` are just `Engine` constructions
(`src/susurro/engine.py:22`) differing only by model name; both run int8 on CUDA,
so turbo-vs-large isolates the *model*, not the precision. The adapter wraps an
`Engine` behind the `Transcriber` protocol and delegates `transcribe` straight to
it — the offline audio path is already built and tested.

Importing this module is still cheap: `Engine.__init__` lazy-imports
faster-whisper, so the CTranslate2/CUDA cost lands only when a backend is actually
built (i.e. when a run selects it), never at collection or `--list-models` time.
"""

from __future__ import annotations

import numpy as np

from ..engine import Engine


class FasterWhisperTranscriber:
    """A `Transcriber` backed by a warm faster-whisper `Engine` (int8/CUDA)."""

    def __init__(self, model_name: str, *, device: str = "cuda") -> None:
        # Build the model warm in __init__, like Engine/the daemon: the runner
        # measures this as informational load time, separate from per-clip latency.
        # Engine's default RuleBasedFormatter (whitespace trim/collapse only) is
        # exactly the "what gets typed" text we want to score; the scorer owns the
        # heavier WER normalization (Step 4), so no extra cleanup belongs here.
        self._engine = Engine(model_name, device=device, compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        return self._engine.transcribe(audio)
