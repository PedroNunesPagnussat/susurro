"""Rule-based text cleanup: the seam between what Whisper produced and what gets
typed.

`Formatter` is a protocol so a smarter impl (e.g. a local-LLM cleanup pass) can
drop in behind the same interface without touching the Engine. The only impl today
trims/collapses whitespace and drops empty / no-speech results.

Deliberately **no filler-word removal**: regex eats real words ("the sum" is not
filler); that belongs in a future LLM stage, not a blunt rule.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WHITESPACE = re.compile(r"\s+")


@runtime_checkable
class Formatter(Protocol):
    """Turns a raw transcript into the text to emit; "" means emit nothing."""

    def format(self, text: str | None) -> str: ...


class RuleBasedFormatter:
    """Trim + collapse whitespace; drop empty / no-speech results. Nothing else."""

    def format(self, text: str | None) -> str:
        if not text:
            return ""
        # Collapse every run of whitespace (spaces, newlines, tabs) to one space,
        # then strip the ends. A now-empty string signals "nothing to emit".
        return _WHITESPACE.sub(" ", text).strip()
