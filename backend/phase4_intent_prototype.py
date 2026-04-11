from __future__ import annotations

from phase4_node_engine import (
    ParseResult,
    TransitionDecision,
    advance_node,
    load_node_spec,
    normalize_text,
    parse_hypnose_progress,
)


def classify_hypnose_progress(value: str) -> ParseResult:
    return parse_hypnose_progress(value)


def advance_hell_hypnose_pause(user_text: str) -> TransitionDecision:
    return advance_node("hell_hypnose_pause", user_text)


__all__ = [
    "ParseResult",
    "TransitionDecision",
    "advance_hell_hypnose_pause",
    "classify_hypnose_progress",
    "load_node_spec",
    "normalize_text",
]
