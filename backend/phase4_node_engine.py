from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "docs" / "phase4_node_catalog_prototype"


@dataclass(frozen=True)
class ParseResult:
    intent: str
    confidence: float
    normalized_text: str
    reason: str
    needs_clarification: bool


@dataclass(frozen=True)
class TransitionDecision:
    node_id: str
    next_node: str
    reply_text: str
    parse_result: ParseResult


def load_node_spec(node_id: str) -> dict[str, Any]:
    path = SPEC_DIR / f"{node_id}.json"
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    _validate_node_spec(spec, node_id)
    return spec


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    replacements = {
        "\u00e4": "ae",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
        "\u00c3\u00a4": "ae",
        "\u00c3\u00b6": "oe",
        "\u00c3\u00bc": "ue",
        "\u00c3\u009f": "ss",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    token_aliases = {
        "angehnem": "angenehm",
        "unangehnem": "unangenehm",
        "auf geloest": "aufgeloest",
        "loestsich": "loest sich",
    }
    for src, dest in token_aliases.items():
        text = text.replace(src, dest)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_any(text: str, cues: tuple[str, ...]) -> str | None:
    text_tokens = text.split()
    for cue in cues:
        cue_tokens = cue.split()
        if len(cue_tokens) == 1 and cue_tokens[0] in text_tokens:
            return cue
        if len(cue_tokens) > 1 and _contains_tokens_in_order(text_tokens, cue_tokens):
            return cue

    return None


def _contains_tokens_in_order(text_tokens: list[str], cue_tokens: list[str]) -> bool:
    if not cue_tokens:
        return False
    start_index = 0
    for cue_token in cue_tokens:
        try:
            match_index = text_tokens.index(cue_token, start_index)
        except ValueError:
            return False
        start_index = match_index + 1
    return True


def _validate_node_spec(spec: dict[str, Any], expected_node_id: str) -> None:
    required_keys = (
        "node_id",
        "question_text",
        "intent_family",
        "allowed_intents",
        "fallback_question",
        "transitions",
    )
    missing = [key for key in required_keys if key not in spec]
    if missing:
        raise ValueError(
            f"Node spec '{expected_node_id}' is missing required keys: {', '.join(missing)}"
        )

    if spec["node_id"] != expected_node_id:
        raise ValueError(
            f"Node spec id mismatch: expected '{expected_node_id}', got '{spec['node_id']}'"
        )

    allowed_intents = tuple(spec["allowed_intents"])
    transitions = spec["transitions"]
    missing_transitions = [intent for intent in allowed_intents if intent not in transitions]
    if missing_transitions:
        raise ValueError(
            f"Node spec '{expected_node_id}' is missing transitions for: "
            f"{', '.join(missing_transitions)}"
        )

    response_intents = [
        intent for intent in allowed_intents if intent not in {"unclear", "repeat", "abort"}
    ]
    responses = spec.get("responses", {})
    missing_responses = [intent for intent in response_intents if intent not in responses]
    if missing_responses:
        raise ValueError(
            f"Node spec '{expected_node_id}' is missing responses for: "
            f"{', '.join(missing_responses)}"
        )


REPEAT_CUES = (
    "wiederhole",
    "wiederholen",
    "nochmal",
    "noch mal",
    "noch einmal",
)

ABORT_CUES = (
    "abbrechen",
    "beenden",
    "stop",
    "schluss",
    "ich will raus",
)

UNCLEAR_CUES = (
    "weiss nicht",
    "nicht sicher",
    "schwer zu sagen",
    "unklar",
    "kann ich nicht einordnen",
    "nicht klar",
)


def _base_parse(text: str) -> ParseResult | None:
    cue = _contains_any(text, ABORT_CUES)
    if cue:
        return ParseResult("abort", 0.98, text, f"matched abort cue '{cue}'", False)

    cue = _contains_any(text, REPEAT_CUES)
    if cue:
        return ParseResult("repeat", 0.96, text, f"matched repeat cue '{cue}'", False)

    cue = _contains_any(text, UNCLEAR_CUES)
    if cue:
        return ParseResult("unclear", 0.93, text, f"matched unclear cue '{cue}'", True)

    return None


NEGATED_RESOLVED_CUES = (
    "noch nicht aufgeloest",
    "noch nicht geloest",
    "nicht aufgeloest",
    "nicht geloest",
    "noch nicht weg",
)

RESOLVED_CUES = (
    "bereits aufgeloest",
    "schon aufgeloest",
    "ist aufgeloest",
    "aufgeloest",
    "hat sich geloest",
    "ist geloest",
    "geloest",
    "ist weg",
    "weg",
    "fertig",
    "vorbei",
)

RESOLVING_CUES = (
    "loest sich auf",
    "ist noch im loesen",
    "noch im loesen",
    "loest sich noch",
    "arbeitet noch",
    "passiert noch etwas",
    "ist am arbeiten",
)

NEED_MORE_TIME_CUES = (
    "brauche noch einen moment",
    "brauch noch einen moment",
    "gib mir noch einen moment",
    "noch etwas zeit",
    "noch kurz zeit",
    "noch einen moment",
    "noch nicht ganz",
    "etwas zeit",
)

PLEASANT_CUES = (
    "fuehlt sich angenehm an",
    "sehr angenehm",
    "ganz angenehm",
    "eher angenehm",
    "angenehm",
    "ruhig",
    "gut",
    "leicht",
    "warm",
    "friedlich",
)

UNPLEASANT_CUES = (
    "fuehlt sich unangenehm an",
    "eher unangenehm",
    "unangenehm",
    "schlecht",
    "stress",
    "druck",
    "angst",
    "komisch",
    "ungut",
    "drueckend",
)

KNOWN_CUES = (
    "schon bekannt",
    "kenne ich schon",
    "kenne ich",
    "von frueher",
    "frueher schon",
    "schon mal",
    "schon einmal",
    "bereits erlebt",
    "hatte ich schon",
)

NEW_CUES = (
    "zum ersten mal",
    "erstes mal",
    "das erste mal",
    "neu",
    "noch nie",
    "kannte ich nicht",
    "hatte ich davor nicht",
    "vorher nicht",
)


def parse_hypnose_progress(value: str) -> ParseResult:
    text = normalize_text(value)
    if not text:
        return ParseResult("unclear", 0.0, text, "empty input", True)

    base = _base_parse(text)
    if base is not None:
        return base

    cue = _contains_any(text, NEGATED_RESOLVED_CUES)
    if cue:
        return ParseResult("need_more_time", 0.9, text, f"matched negated resolved cue '{cue}'", False)

    cue = _contains_any(text, RESOLVING_CUES)
    if cue:
        return ParseResult("resolving", 0.9, text, f"matched resolving cue '{cue}'", False)

    cue = _contains_any(text, NEED_MORE_TIME_CUES)
    if cue:
        return ParseResult("need_more_time", 0.88, text, f"matched need-more-time cue '{cue}'", False)

    cue = _contains_any(text, RESOLVED_CUES)
    if cue:
        return ParseResult("resolved", 0.95, text, f"matched resolved cue '{cue}'", False)

    return ParseResult("unclear", 0.28, text, "no matching cue cluster", True)


def parse_pleasantness(value: str) -> ParseResult:
    text = normalize_text(value)
    if not text:
        return ParseResult("unclear", 0.0, text, "empty input", True)

    base = _base_parse(text)
    if base is not None:
        return base

    cue = _contains_any(text, UNPLEASANT_CUES)
    if cue:
        return ParseResult("unpleasant", 0.92, text, f"matched unpleasant cue '{cue}'", False)

    cue = _contains_any(text, PLEASANT_CUES)
    if cue:
        return ParseResult("pleasant", 0.93, text, f"matched pleasant cue '{cue}'", False)

    return ParseResult("unclear", 0.3, text, "no pleasantness cue matched", True)


def parse_known_vs_new(value: str) -> ParseResult:
    text = normalize_text(value)
    if not text:
        return ParseResult("unclear", 0.0, text, "empty input", True)

    base = _base_parse(text)
    if base is not None:
        return base

    cue = _contains_any(text, NEW_CUES)
    if cue:
        return ParseResult("new", 0.93, text, f"matched new cue '{cue}'", False)

    cue = _contains_any(text, KNOWN_CUES)
    if cue:
        return ParseResult("known", 0.92, text, f"matched known cue '{cue}'", False)

    return ParseResult("unclear", 0.31, text, "no known/new cue matched", True)


PARSERS: dict[str, Callable[[str], ParseResult]] = {
    "hypnose_progress": parse_hypnose_progress,
    "pleasantness": parse_pleasantness,
    "known_vs_new": parse_known_vs_new,
}


def advance_node(node_id: str, user_text: str) -> TransitionDecision:
    spec = load_node_spec(node_id)
    family = spec["intent_family"]
    parser = PARSERS[family]
    parsed = parser(user_text)
    next_node = spec["transitions"][parsed.intent]

    if parsed.intent == "repeat":
        reply_text = spec["question_text"]
    elif parsed.intent == "abort":
        reply_text = spec.get("abort_question", "Moechtest du die Sitzung wirklich abbrechen?")
    elif parsed.intent == "unclear":
        reply_text = spec["fallback_question"]
    else:
        reply_text = spec["responses"][parsed.intent]

    return TransitionDecision(
        node_id=node_id,
        next_node=next_node,
        reply_text=reply_text,
        parse_result=parsed,
    )

