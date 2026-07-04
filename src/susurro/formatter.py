"""Rule-based cleanup formatter — the Phase-1 formatter stage.

The formatter is a pluggable seam from day one (`Formatter` protocol) so Phase 2
can drop in a local-LLM cleanup impl behind the same interface without touching
the Engine. Phase 1's only impl trims/collapses whitespace and drops empty /
no-speech results.

Deliberately **no filler-word removal**: regex eats real words ("the sum" is not
filler), so it's deferred to the Phase-2 LLM stage (see plan Decisions).
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
