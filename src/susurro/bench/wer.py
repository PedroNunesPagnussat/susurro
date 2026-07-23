"""Accuracy scoring: normalized WER (headline), raw WER, and CER for one
(reference, hypothesis) pair, via `jiwer`.

Three numbers because each answers a different question (see spec Decisions):
- **normalized WER** — lowercase + strip punctuation + collapse whitespace. The
  comparable headline: same number across models and against published benchmarks.
- **raw WER** — case and punctuation preserved (jiwer's default transform). Closest
  to lived quality: it's what actually gets typed into the window.
- **CER** — character error rate on the normalized text. A normalization-robust
  tiebreaker that de-noises word-boundary and tokenization quirks.

Deliberately no number-word normalization: `jiwer` has none and pulling in
openai-whisper's normalizer would drag in torch. The scripts avoid ambiguous
digit/word number forms instead — a documented sensitivity of these numbers.

`jiwer` is a `bench`-extra dependency, so this module is imported lazily by the
runner (never at `import susurro.bench` / collection time).
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

# Shared normalization prefix for the comparable metrics. jiwer's own transforms,
# no external normalizer.
_NORMALIZE = [
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
]
# WER needs word lists; CER needs char lists — same normalization, different reduce.
_NORM_WORDS = jiwer.Compose([*_NORMALIZE, jiwer.ReduceToListOfListOfWords()])
_NORM_CHARS = jiwer.Compose([*_NORMALIZE, jiwer.ReduceToListOfListOfChars()])


@dataclass(frozen=True)
class Score:
    """The three accuracy numbers for one clip. `norm_wer` is the headline."""

    norm_wer: float
    raw_wer: float
    cer: float


def score(reference: str, hypothesis: str) -> Score:
    """Score `hypothesis` against `reference`. Lower is better; 0.0 is perfect.

    Raw WER uses jiwer's default transform (case + punctuation preserved); the
    normalized WER and CER apply the shared lowercase/strip-punctuation/collapse
    normalization first."""
    return Score(
        norm_wer=jiwer.wer(
            reference,
            hypothesis,
            reference_transform=_NORM_WORDS,
            hypothesis_transform=_NORM_WORDS,
        ),
        raw_wer=jiwer.wer(reference, hypothesis),
        cer=jiwer.cer(
            reference,
            hypothesis,
            reference_transform=_NORM_CHARS,
            hypothesis_transform=_NORM_CHARS,
        ),
    )
