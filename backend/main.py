from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
import html
import uuid
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from config.tts_profiles import get_tts_profile

try:
    from google.cloud import texttospeech
except Exception:
    texttospeech = None

DEFAULT_MODEL_ID = "ft:gpt-3.5-turbo-1106:personal::AzSLcCUs"
MAX_HISTORY_MESSAGES = 24
PHASE4_MODEL_ID = "phase4_orchestrator_v1"
PHASE4_MAX_BACKTRACE_LOOPS = 2
DEFAULT_PHASE4_PROMPTS_PATH = (
    r"C:\Projekte\hypnose_systemV2\flow_studio\catalog\phase4\phase4_prompts.json"
)
SYSTEM_PROMPT = (
    "Du bist eine ruhige, empathische Hypnose-Begleiterin. "
    "Antworte klar, freundlich und kurz genug fuer gesprochene Ausgabe."
)
PHASE4_DEFAULT_PROMPTS: dict[str, str] = {
    "phase4_intro_room": (
        "Du bist jetzt im Raum der Veraenderung und findest den Weg zur Ursprungsursache."
    ),
    "phase4_activation_prepare": (
        "Ruf jetzt das aktivierte Gefuehl noch einmal klar in dir auf und verstaerke es."
    ),
    "phase4_activation_ready_question": "Spuerst du das Gefuehl jetzt stark genug?",
    "phase4_activation_not_ready_followup": (
        "Spuerst du gar nichts mehr oder ist noch ein kleiner Rest vorhanden?"
    ),
    "phase4_activation_not_ready_remainder": (
        "Gut. Bleib noch kurz dabei und gib Bescheid, sobald es wieder klar spuerbar ist."
    ),
    "phase4_activation_not_ready_none": (
        "Gut. Dann gehen wir nicht ueber dieses Gefuehl zurueck."
    ),
    "phase4_activation_first_cigarette_confirm": "Ist das fuer dich so stimmig?",
    "phase4_activation_first_cigarette_clarify": (
        "Sag mir bitte kurz, was noch unklar ist."
    ),
    "phase4_activation_countdown": (
        "Ich zaehle jetzt von 5 bis 0 und bei 0 gehst du zum Ursprung."
    ),
    "phase4_activation_first_cigarette_countdown": (
        "Ich zaehle von 5 bis 0 und bei 0 landest du beim ersten Kontakt mit Rauchen."
    ),
    "phase4_after_snap_arrival": "Und jetzt bist du genau dort. Am Ursprung.",
    "phase4_after_snap_arrival_first_cigarette": (
        "Und jetzt bist du im fruehen Moment des ersten Kontakts mit Rauchen."
    ),
    "phase4_known_question": "Hast du dieses Gefuehl davor schon einmal erlebt?",
    "phase4_known_timeout_fallback": (
        "Dann gehen wir sicherheitshalber noch einen Schritt weiter zurueck."
    ),
    "phase4_closing_origin_scene": (
        "Gut. Dann bleiben wir in genau dieser Szene als Ursprung."
    ),
    "phase4_origin_scene_reflection": (
        "Du bist also jetzt an einem Moment, in dem du {age} bist, {scene} wahrnimmst und das Gefuehl sich als {feeling} zeigt."
    ),
    "entry_perception_light": "Was nimmst du dort wahr, eher hell oder eher dunkel?",
    "entry_clarify_again": (
        "Was nimmst du jetzt wahr: eher hell, eher dunkel oder beides?"
    ),
    "entry_map_both_to_dark": (
        "Gut. Dann folgen wir dem dunkleren Anteil, dort liegt meist der emotionale Kern."
    ),
    "dark_known_or_new_question": (
        "Fuehlt sich dieses ungute Gefuehl fuer dich eher vertraut an oder eher neu?"
    ),
    "dark_known_or_new_retry": (
        "Antworte bitte kurz mit: vertraut oder neu."
    ),
    "dark_backtrace_countdown": (
        "Gut. Dann gehen wir noch weiter zurueck. Ich zaehle 5 bis 0."
    ),
    "dark_backtrace_arrival": "Gut. Du bist jetzt an einem frueheren Punkt angekommen.",
    "dark_loop_guard_reached": (
        "Gut. Wir haben den fruehesten erreichbaren Ursprung jetzt eingegrenzt."
    ),
    "dark_origin_reached": "Gut. Dann sind wir jetzt am Ursprung.",
    "dark_origin_age_question": "Wie alt bist du dort ganz spontan?",
    "hell_bright_followup": (
        "Wie hell ist es gerade: sehr hell, oder kannst du beim Umschauen noch etwas wahrnehmen?"
    ),
    "hell_feel_question": (
        "Wie fuehlt sich dieses Helle fuer dich an: eher angenehm oder eher unangenehm?"
    ),
    "hell_hypnose_loch_notice": (
        "Sehr gut. Bleib kurz in diesem Zustand und lass den Prozess arbeiten."
    ),
    "hell_hypnose_loch_open_followup": (
        "Wie fuehlt es sich jetzt an: loest es sich noch auf oder ist es bereits geloest?"
    ),
    "hell_unpleasant_regulation_intro": (
        "Gut. Dann regulieren wir das jetzt mit mehr Distanz und weniger Intensitaet."
    ),
    "hell_unpleasant_regulation_choice": (
        "Was hilft dir gerade am meisten: mehr Abstand, weniger Helligkeit oder klarerer Fokus?"
    ),
    "hell_unpleasant_regulation_check": (
        "Wie wirkt die Szene jetzt: eher hell, eher dunkel, beides oder ruhiger?"
    ),
    "hell_post_resolved_continue": "Gut. Dann gehen wir jetzt strukturiert weiter.",
    "hell_stay_extra_time": "Nimm dir noch einen Moment Zeit und gib mir dann kurz Bescheid.",
}

PAUSE_PRESETS_MS = {
    "kurz": 320,
    "mittel": 700,
    "lang": 1200,
    "short": 320,
    "medium": 700,
    "long": 1200,
}

PAUSE_MARKER_PATTERN = re.compile(
    r"(?:\[\[|\[)\s*"
    r"(?P<tag>kurz|mittel|lang|short|medium|long|pause)"
    r"(?:\s*:\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>ms|s)?)?"
    r"\s*(?:\]\]|\])",
    re.IGNORECASE,
)

ANY_MARKER_PATTERN = re.compile(r"(?:\[\[|\[)\s*[^\]\n]{1,64}\s*(?:\]\]|\])")

load_dotenv()

app = FastAPI(title="Hypnose Chat Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=6000)
    reset_session: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    awaits_user_input: bool = False
    phase4_active: bool = False
    phase4_node: str | None = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class Phase4StartRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class TtsPayload(BaseModel):
    text: str = Field(min_length=1, max_length=24000)
    context: str = Field(default="session_phase_4", max_length=80)
    phase: int | None = None


_history_lock = threading.Lock()
_session_history: dict[str, deque[dict[str, str]]] = {}
_phase4_state: dict[str, dict[str, object]] = {}


def _load_phase4_prompts() -> dict[str, str]:
    merged = dict(PHASE4_DEFAULT_PROMPTS)
    source_path = Path(
        (os.getenv("PHASE4_PROMPTS_SOURCE_PATH") or DEFAULT_PHASE4_PROMPTS_PATH).strip()
        or DEFAULT_PHASE4_PROMPTS_PATH
    )
    if not source_path.exists():
        return merged
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return merged
    except Exception:
        return merged
    for key, value in payload.items():
        clean_key = str(key).strip()
        clean_value = str(value or "").strip()
        if clean_key and clean_value:
            merged[clean_key] = clean_value
    return merged


PHASE4_PROMPTS = _load_phase4_prompts()


def _phase4_prompt(key: str) -> str:
    return str(PHASE4_PROMPTS.get(key) or PHASE4_DEFAULT_PROMPTS.get(key) or key)


def _normalize_text(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "\u00e4": "ae",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_yes_no(value: str) -> bool | None:
    text = _normalize_text(value)
    if not text:
        return None
    yes_tokens = {"ja", "jap", "jup", "klar", "ok", "okay", "passt", "stimmt", "genau"}
    no_tokens = {"nein", "noe", "nee", "ne", "nicht", "kein", "nichts"}
    tokens = set(text.split())
    has_yes = bool(tokens & yes_tokens) or any(cue in text for cue in ("ja", "stimmt", "passt", "genau"))
    has_no = bool(tokens & no_tokens) or any(
        cue in text for cue in ("nein", "gar nicht", "nicht mehr", "keins", "nichts")
    )
    if has_yes and has_no:
        return None
    if has_yes:
        return True
    if has_no:
        return False
    return None


def _classify_scene_light(value: str) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if "beides" in text or ("hell" in text and "dunkel" in text):
        return "beides"
    if any(cue in text for cue in ("dunkel", "schwarz", "nacht", "finster")):
        return "dunkel"
    if any(cue in text for cue in ("hell", "licht", "weiss", "blend")):
        return "hell"
    return None


def _classify_known_or_new(value: str) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if any(cue in text for cue in ("vertraut", "kenne", "schon", "frueher", "fruher")):
        return "known"
    if any(cue in text for cue in ("neu", "erstmals", "erstes mal", "noch nie")):
        return "new"
    return None


def _classify_pleasant(value: str) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if any(cue in text for cue in ("unangenehm", "schlecht", "druck", "stress", "angst")):
        return "unpleasant"
    if any(cue in text for cue in ("angenehm", "ruhig", "gut", "leicht", "warm")):
        return "pleasant"
    return None


def _build_phase4_reply(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if str(part or "").strip())


def _phase4_build_response(
    *,
    session_id: str,
    reply: str,
    node: str,
    awaits_user_input: bool,
    active: bool,
) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        model=PHASE4_MODEL_ID,
        awaits_user_input=awaits_user_input,
        phase4_active=active,
        phase4_node=node,
    )


def _phase4_initial_reply() -> str:
    return _build_phase4_reply(
        [
            _phase4_prompt("phase4_intro_room"),
            _phase4_prompt("phase4_activation_prepare"),
            _phase4_prompt("phase4_activation_ready_question"),
        ]
    )


def _phase4_advance(session_id: str, message: str) -> ChatResponse:
    with _history_lock:
        state = dict(_phase4_state.get(session_id) or {})
        if not state or not bool(state.get("active")):
            raise HTTPException(status_code=409, detail="Phase 4 ist fuer diese Session nicht aktiv.")

        node = str(state.get("node") or "activation_ready")
        loops = int(state.get("backtrace_loops") or 0)
        normalized = _normalize_text(message)

        next_node = node
        awaits_user_input = True
        active = True
        reply = ""

        if node == "activation_ready":
            yes_no = _parse_yes_no(message)
            if yes_no is True:
                next_node = "entry_light"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("phase4_activation_countdown"),
                        _phase4_prompt("phase4_after_snap_arrival"),
                        _phase4_prompt("entry_perception_light"),
                    ]
                )
            elif yes_no is False:
                next_node = "activation_not_ready_followup"
                reply = _phase4_prompt("phase4_activation_not_ready_followup")
            else:
                reply = _phase4_prompt("phase4_activation_ready_question")

        elif node == "activation_not_ready_followup":
            no_signal = any(
                cue in normalized
                for cue in ("gar nicht", "nicht mehr", "nichts", "weg", "null", "kein")
            )
            if no_signal:
                next_node = "first_cig_confirm"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("phase4_activation_not_ready_none"),
                        _phase4_prompt("phase4_activation_first_cigarette_confirm"),
                    ]
                )
            else:
                next_node = "activation_ready"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("phase4_activation_not_ready_remainder"),
                        _phase4_prompt("phase4_activation_ready_question"),
                    ]
                )

        elif node == "first_cig_confirm":
            yes_no = _parse_yes_no(message)
            if yes_no is True:
                next_node = "known_question"
                loops = 0
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("phase4_activation_first_cigarette_countdown"),
                        _phase4_prompt("phase4_after_snap_arrival_first_cigarette"),
                        _phase4_prompt("phase4_known_question"),
                    ]
                )
            elif yes_no is False:
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("phase4_activation_first_cigarette_clarify"),
                        _phase4_prompt("phase4_activation_first_cigarette_confirm"),
                    ]
                )
            else:
                reply = _phase4_prompt("phase4_activation_first_cigarette_confirm")

        elif node == "entry_light":
            light = _classify_scene_light(message)
            if light == "hell":
                next_node = "hell_bright_followup"
                reply = _phase4_prompt("hell_bright_followup")
            elif light in {"dunkel", "beides"}:
                next_node = "dark_known"
                parts = []
                if light == "beides":
                    parts.append(_phase4_prompt("entry_map_both_to_dark"))
                parts.append(_phase4_prompt("dark_known_or_new_question"))
                reply = _build_phase4_reply(parts)
            else:
                reply = _phase4_prompt("entry_clarify_again")

        elif node == "hell_bright_followup":
            light = _classify_scene_light(message)
            very_bright = any(cue in normalized for cue in ("sehr hell", "blend", "weiss", "nur licht"))
            if very_bright:
                next_node = "hell_feel"
                reply = _phase4_prompt("hell_feel_question")
            elif light in {"dunkel", "beides"}:
                next_node = "dark_known"
                parts = []
                if light == "beides":
                    parts.append(_phase4_prompt("entry_map_both_to_dark"))
                parts.append(_phase4_prompt("dark_known_or_new_question"))
                reply = _build_phase4_reply(parts)
            else:
                reply = _phase4_prompt("hell_bright_followup")

        elif node == "hell_feel":
            pleasant = _classify_pleasant(message)
            if pleasant == "pleasant":
                next_node = "hell_hypnose_followup"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("hell_hypnose_loch_notice"),
                        _phase4_prompt("hell_hypnose_loch_open_followup"),
                    ]
                )
            elif pleasant == "unpleasant":
                next_node = "hell_regulation_choice"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("hell_unpleasant_regulation_intro"),
                        _phase4_prompt("hell_unpleasant_regulation_choice"),
                    ]
                )
            else:
                reply = _phase4_prompt("hell_feel_question")

        elif node == "hell_hypnose_followup":
            resolved = any(
                cue in normalized
                for cue in ("geloest", "aufgeloest", "aufgeloest", "ruhig", "fertig", "besser", "ja")
            )
            if resolved:
                next_node = "dark_known"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("hell_post_resolved_continue"),
                        _phase4_prompt("dark_known_or_new_question"),
                    ]
                )
            else:
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("hell_stay_extra_time"),
                        _phase4_prompt("hell_hypnose_loch_open_followup"),
                    ]
                )

        elif node == "hell_regulation_choice":
            next_node = "hell_regulation_check"
            reply = _phase4_prompt("hell_unpleasant_regulation_check")

        elif node == "hell_regulation_check":
            light = _classify_scene_light(message)
            if light == "hell":
                next_node = "hell_feel"
                reply = _phase4_prompt("hell_feel_question")
            elif light in {"dunkel", "beides"} or "ruhig" in normalized:
                next_node = "dark_known"
                parts = []
                if light == "beides":
                    parts.append(_phase4_prompt("entry_map_both_to_dark"))
                parts.append(_phase4_prompt("dark_known_or_new_question"))
                reply = _build_phase4_reply(parts)
            else:
                reply = _phase4_prompt("hell_unpleasant_regulation_check")

        elif node == "dark_known":
            known_state = _classify_known_or_new(message)
            if known_state == "known":
                loops += 1
                if loops >= PHASE4_MAX_BACKTRACE_LOOPS:
                    next_node = "origin_age"
                    reply = _build_phase4_reply(
                        [
                            _phase4_prompt("dark_loop_guard_reached"),
                            _phase4_prompt("dark_origin_reached"),
                            _phase4_prompt("dark_origin_age_question"),
                        ]
                    )
                else:
                    next_node = "dark_known"
                    reply = _build_phase4_reply(
                        [
                            _phase4_prompt("dark_backtrace_countdown"),
                            _phase4_prompt("dark_backtrace_arrival"),
                            _phase4_prompt("dark_known_or_new_question"),
                        ]
                    )
            elif known_state == "new":
                next_node = "origin_age"
                reply = _build_phase4_reply(
                    [
                        _phase4_prompt("dark_origin_reached"),
                        _phase4_prompt("dark_origin_age_question"),
                    ]
                )
            else:
                reply = _phase4_prompt("dark_known_or_new_retry")

        elif node == "known_question":
            yes_no = _parse_yes_no(message)
            if yes_no is True:
                loops += 1
                if loops >= PHASE4_MAX_BACKTRACE_LOOPS:
                    next_node = "origin_age"
                    reply = _build_phase4_reply(
                        [
                            _phase4_prompt("phase4_known_timeout_fallback"),
                            _phase4_prompt("dark_origin_age_question"),
                        ]
                    )
                else:
                    next_node = "known_question"
                    reply = _build_phase4_reply(
                        [
                            _phase4_prompt("phase4_known_timeout_fallback"),
                            _phase4_prompt("phase4_known_question"),
                        ]
                    )
            elif yes_no is False:
                next_node = "origin_age"
                reply = _phase4_prompt("dark_origin_age_question")
            else:
                reply = _phase4_prompt("phase4_known_question")

        elif node == "origin_age":
            age = str(message or "").strip() or "[unklar]"
            reflection = _phase4_prompt("phase4_origin_scene_reflection").format(
                age=age,
                scene="diese Szene",
                feeling="deutlich",
            )
            next_node = "completed"
            awaits_user_input = False
            active = False
            reply = _build_phase4_reply(
                [
                    _phase4_prompt("phase4_closing_origin_scene"),
                    reflection,
                ]
            )
        else:
            next_node = "activation_ready"
            reply = _phase4_prompt("phase4_activation_ready_question")

        state["node"] = next_node
        state["backtrace_loops"] = loops
        state["active"] = active
        _session_history.setdefault(session_id, deque(maxlen=MAX_HISTORY_MESSAGES)).append(
            {"role": "user", "content": message}
        )
        _session_history.setdefault(session_id, deque(maxlen=MAX_HISTORY_MESSAGES)).append(
            {"role": "assistant", "content": reply}
        )
        if active:
            _phase4_state[session_id] = state
        else:
            _phase4_state.pop(session_id, None)

    return _phase4_build_response(
        session_id=session_id,
        reply=reply,
        node=next_node,
        awaits_user_input=awaits_user_input,
        active=active,
    )


def _resolve_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_key") or "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY fehlt im Backend.")
    return key


def _resolve_model_id() -> str:
    return (os.getenv("HYPNOSE_MODEL_ID") or DEFAULT_MODEL_ID).strip()


def _resolve_openai_client() -> OpenAI:
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    return OpenAI(api_key=_resolve_api_key(), base_url=base_url)


def _extract_reply_text(raw_completion: object) -> str:
    try:
        choices = getattr(raw_completion, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return (getattr(message, "content", None) or "").strip()
    except Exception:
        return ""


def _pause_ms_from_marker(match: re.Match[str]) -> int:
    tag = (match.group("tag") or "").strip().lower()
    if tag in PAUSE_PRESETS_MS:
        return PAUSE_PRESETS_MS[tag]
    if tag != "pause":
        return PAUSE_PRESETS_MS["mittel"]

    raw_value = (match.group("value") or "").strip().replace(",", ".")
    raw_unit = (match.group("unit") or "ms").strip().lower()
    try:
        numeric = float(raw_value)
    except Exception:
        return PAUSE_PRESETS_MS["mittel"]

    millis = int(round(numeric * (1000 if raw_unit == "s" else 1)))
    return max(120, min(millis, 8000))


def _segment_text_to_ssml(
    segment: str,
    sentence_break_ms: int,
    comma_break_ms: int,
    ellipsis_break_ms: int,
) -> str:
    source = ANY_MARKER_PATTERN.sub(" ", segment or "")
    source = re.sub(r"\s+", " ", source).strip()
    if not source:
        return ""

    chunks: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        if source.startswith("...", i):
            j = i + 3
            while j < n and source[j] == ".":
                j += 1
            chunks.append(html.escape(source[i:j]))
            if ellipsis_break_ms > 0:
                chunks.append(f"<break time='{int(ellipsis_break_ms)}ms'/>")
            i = j
            continue

        ch = source[i]
        chunks.append(html.escape(ch))

        prev_ch = source[i - 1] if i > 0 else ""
        next_ch = source[i + 1] if i + 1 < n else ""
        at_boundary = (i + 1 >= n) or next_ch.isspace() or next_ch in "\"')]}»”"
        is_decimal_dot = ch == "." and prev_ch.isdigit() and next_ch.isdigit()

        if at_boundary:
            if ch in ("!", "?") and sentence_break_ms > 0:
                chunks.append(f"<break time='{int(sentence_break_ms)}ms'/>")
            elif ch == "." and (not is_decimal_dot) and sentence_break_ms > 0:
                chunks.append(f"<break time='{int(sentence_break_ms)}ms'/>")
            elif ch in (";", ":") and sentence_break_ms > 0:
                soft_break = max(int(sentence_break_ms * 0.65), comma_break_ms)
                if soft_break > 0:
                    chunks.append(f"<break time='{soft_break}ms'/>")
            elif ch == "," and comma_break_ms > 0:
                chunks.append(f"<break time='{int(comma_break_ms)}ms'/>")

        i += 1

    return "".join(chunks)


def _plain_text_to_ssml(
    text: str,
    newline_break_ms: int,
    paragraph_break_ms: int,
    sentence_break_ms: int,
    comma_break_ms: int,
    ellipsis_break_ms: int,
) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return ""

    chunks: list[str] = []
    index = 0
    length = len(normalized)
    while index < length:
        if normalized[index] == "\n":
            run = 1
            while index + run < length and normalized[index + run] == "\n":
                run += 1
            break_ms = paragraph_break_ms if run > 1 else newline_break_ms
            chunks.append(f"<break time='{break_ms}ms'/>")
            index += run
            continue

        next_break = normalized.find("\n", index)
        if next_break == -1:
            next_break = length
        segment = normalized[index:next_break]
        if segment:
            chunks.append(
                _segment_text_to_ssml(
                    segment,
                    sentence_break_ms=sentence_break_ms,
                    comma_break_ms=comma_break_ms,
                    ellipsis_break_ms=ellipsis_break_ms,
                )
            )
        index = next_break

    return "".join(chunks)


def _build_ssml_text(
    raw_text: str,
    newline_break_ms: int,
    paragraph_break_ms: int,
    sentence_break_ms: int,
    comma_break_ms: int,
    ellipsis_break_ms: int,
) -> str:
    source = (raw_text or "").strip()
    if not source:
        return ""

    chunks: list[str] = []
    cursor = 0
    for match in PAUSE_MARKER_PATTERN.finditer(source):
        if match.start() > cursor:
            chunks.append(
                _plain_text_to_ssml(
                    source[cursor:match.start()],
                    newline_break_ms,
                    paragraph_break_ms,
                    sentence_break_ms,
                    comma_break_ms,
                    ellipsis_break_ms,
                )
            )
        chunks.append(f"<break time='{_pause_ms_from_marker(match)}ms'/>")
        cursor = match.end()

    if cursor < len(source):
        chunks.append(
            _plain_text_to_ssml(
                source[cursor:],
                newline_break_ms,
                paragraph_break_ms,
                sentence_break_ms,
                comma_break_ms,
                ellipsis_break_ms,
            )
        )

    body = "".join(part for part in chunks if part).strip()
    return body or html.escape(source)


def _split_text_for_google_tts(raw_text: str, ssml_builder, max_ssml_bytes: int = 4700) -> list[str]:
    source = str(raw_text or "").strip()
    if not source:
        return []

    tokens = re.findall(r"\S+\s*", source) or [source]
    chunks: list[str] = []
    current = ""

    for token in tokens:
        candidate = f"{current}{token}"
        if current:
            candidate_ssml = ssml_builder(candidate, True, True)
            if len(candidate_ssml.encode("utf-8")) > max_ssml_bytes:
                committed = current.strip()
                if committed:
                    chunks.append(committed)
                current = token
                single_ssml = ssml_builder(current, True, True)
                if len(single_ssml.encode("utf-8")) > max_ssml_bytes:
                    fragment = ""
                    for char in current:
                        probe = f"{fragment}{char}"
                        probe_ssml = ssml_builder(probe, True, True)
                        if fragment and len(probe_ssml.encode("utf-8")) > max_ssml_bytes:
                            chunks.append(fragment.strip())
                            fragment = char
                        else:
                            fragment = probe
                    current = fragment
                continue
        current = candidate

    last_chunk = current.strip()
    if last_chunk:
        chunks.append(last_chunk)
    return chunks


def _ensure_google_tts_credentials() -> None:
    if str(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return

    key_path = str(os.getenv("GOOGLE_TTS_KEY") or "").strip()
    if not key_path:
        return

    resolved = Path(key_path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent / resolved
    if not resolved.exists():
        raise RuntimeError(f"GOOGLE_TTS_KEY not found: {resolved}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)


def _synthesize_mp3(text: str, context: str | None = None, phase: int | None = None) -> bytes:
    if texttospeech is None:
        raise RuntimeError("google cloud texttospeech is not installed")

    _ensure_google_tts_credentials()
    client = texttospeech.TextToSpeechClient()
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="de-DE",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    _profile_name, profile = get_tts_profile(context=context, phase=phase)
    audio_cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=float(profile.get("speaking_rate", 0.9)),
    )
    lead_break_ms = max(180, int(profile.get("lead_break_ms", 220)))
    trail_break_ms = max(140, int(profile.get("trail_break_ms", 260)))
    newline_break_ms = int(profile.get("newline_break_ms", 280))
    paragraph_break_ms = int(profile.get("paragraph_break_ms", 720))
    sentence_break_ms = int(profile.get("sentence_break_ms", 150))
    comma_break_ms = int(profile.get("comma_break_ms", 45))
    ellipsis_break_ms = int(profile.get("ellipsis_break_ms", 300))

    def build_ssml_payload(raw_chunk: str, include_lead: bool, include_trail: bool) -> str:
        body = _build_ssml_text(
            raw_chunk,
            newline_break_ms=newline_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            sentence_break_ms=sentence_break_ms,
            comma_break_ms=comma_break_ms,
            ellipsis_break_ms=ellipsis_break_ms,
        )
        lead_tag = f"<break time='{lead_break_ms}ms'/>" if include_lead else ""
        trail_tag = f"<break time='{trail_break_ms}ms'/>" if include_trail else ""
        return f"<speak>{lead_tag}{body}{trail_tag}</speak>"

    chunks = _split_text_for_google_tts(text, build_ssml_payload)
    if not chunks:
        return b""

    mp3_parts: list[bytes] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        ssml = build_ssml_payload(
            chunk,
            include_lead=(index == 0),
            include_trail=(index == total - 1),
        )
        req_input = texttospeech.SynthesisInput(ssml=ssml)
        response = client.synthesize_speech(
            input=req_input,
            voice=voice_params,
            audio_config=audio_cfg,
        )
        mp3_parts.append(response.audio_content)

    return b"".join(mp3_parts)


def _start_phase4_session(session_id: str) -> ChatResponse:
    clean_session_id = session_id.strip()
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="session_id fehlt.")
    reply = _phase4_initial_reply()
    with _history_lock:
        _session_history[clean_session_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
        _session_history[clean_session_id].append({"role": "assistant", "content": reply})
        _phase4_state[clean_session_id] = {
            "active": True,
            "node": "activation_ready",
            "backtrace_loops": 0,
        }
    return _phase4_build_response(
        session_id=clean_session_id,
        reply=reply,
        node="activation_ready",
        awaits_user_input=True,
        active=True,
    )


def _chat_turn_internal(
    session_id: str,
    message: str,
    *,
    reset_session: bool = False,
) -> ChatResponse:
    clean_session_id = session_id.strip()
    clean_message = message.strip()
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="session_id fehlt.")
    if not clean_message:
        raise HTTPException(status_code=400, detail="message fehlt.")

    with _history_lock:
        phase4_active = bool((_phase4_state.get(clean_session_id) or {}).get("active"))
        if reset_session:
            _session_history[clean_session_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
            _phase4_state.pop(clean_session_id, None)
        history = _session_history.setdefault(
            clean_session_id,
            deque(maxlen=MAX_HISTORY_MESSAGES),
        )
        history_snapshot = list(history)

    if phase4_active:
        return _phase4_advance(clean_session_id, clean_message)

    payload_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history_snapshot,
        {"role": "user", "content": clean_message},
    ]

    model = _resolve_model_id()
    try:
        completion = _resolve_openai_client().chat.completions.create(
            model=model,
            messages=payload_messages,
            temperature=0.7,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI-Request fehlgeschlagen: {exc}",
        ) from exc

    reply = _extract_reply_text(completion)
    if not reply:
        raise HTTPException(
            status_code=502,
            detail="Leere Antwort vom Modell erhalten.",
        )

    with _history_lock:
        history = _session_history.setdefault(
            clean_session_id,
            deque(maxlen=MAX_HISTORY_MESSAGES),
        )
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        session_id=clean_session_id,
        reply=reply,
        model=model,
        awaits_user_input=False,
        phase4_active=False,
        phase4_node=None,
    )


def _is_phase4_active(session_id: str) -> bool:
    clean_session_id = session_id.strip()
    if not clean_session_id:
        return False
    with _history_lock:
        return bool((_phase4_state.get(clean_session_id) or {}).get("active"))


def _should_start_phase4_from_voice(session_id: str, text: str) -> bool:
    if _is_phase4_active(session_id):
        return False
    normalized = _normalize_text(text)
    if not normalized:
        return False
    wants_start = any(token in normalized for token in ("start", "starte", "beginn"))
    mentions_session = (
        "hypnose" in normalized
        or "session" in normalized
        or "phase 4" in normalized
        or "phase4" in normalized
        or "raum der veraenderung" in normalized
    )
    return wants_start and mentions_session


@dataclass
class VoiceLoopSession:
    session_id: str
    listen_connections: set[WebSocket] = field(default_factory=set)
    speak_connections: set[WebSocket] = field(default_factory=set)
    audio_chunks: list[bytes] = field(default_factory=list)
    audio_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: str = "idle"

    def is_empty(self) -> bool:
        return not self.listen_connections and not self.speak_connections


class VoiceLoopManager:
    def __init__(self) -> None:
        self._sessions: dict[str, VoiceLoopSession] = {}
        self._sessions_lock = asyncio.Lock()

    async def _get_or_create(self, session_id: str) -> VoiceLoopSession:
        clean_session_id = session_id.strip()
        async with self._sessions_lock:
            session = self._sessions.get(clean_session_id)
            if session is None:
                session = VoiceLoopSession(session_id=clean_session_id)
                self._sessions[clean_session_id] = session
            return session

    async def _remove_if_empty(self, session: VoiceLoopSession) -> None:
        if not session.is_empty():
            return
        async with self._sessions_lock:
            existing = self._sessions.get(session.session_id)
            if existing is session and existing.is_empty():
                self._sessions.pop(session.session_id, None)

    async def _send_json_safe(self, websocket: WebSocket, payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            return False

    async def _broadcast(self, targets: set[WebSocket], payload: dict) -> None:
        dead: list[WebSocket] = []
        for connection in list(targets):
            ok = await self._send_json_safe(connection, payload)
            if not ok:
                dead.append(connection)
        for connection in dead:
            targets.discard(connection)

    async def _broadcast_state(self, session: VoiceLoopSession, status: str) -> None:
        session.state = status
        payload = {"type": "state", "status": status}
        await self._broadcast(session.listen_connections, payload)
        await self._broadcast(session.speak_connections, payload)

    def _assistant_payload(self, response: ChatResponse) -> dict:
        return {
            "type": "assistant_text",
            "text": response.reply,
            "isFinal": True,
            "awaits_user_input": response.awaits_user_input,
            "phase4_active": response.phase4_active,
            "phase4_node": response.phase4_node,
            "model": response.model,
            "session_id": response.session_id,
        }

    async def register_listener(self, session_id: str, websocket: WebSocket) -> None:
        session = await self._get_or_create(session_id)
        session.listen_connections.add(websocket)
        await self._broadcast_state(session, session.state)

    async def unregister_listener(self, session_id: str, websocket: WebSocket) -> None:
        session = await self._get_or_create(session_id)
        session.listen_connections.discard(websocket)
        await self._remove_if_empty(session)

    async def register_speaker(self, session_id: str, websocket: WebSocket) -> None:
        session = await self._get_or_create(session_id)
        session.speak_connections.add(websocket)
        await self._send_json_safe(websocket, {"type": "ready"})
        await self._send_json_safe(websocket, {"type": "state", "status": session.state})

    async def unregister_speaker(self, session_id: str, websocket: WebSocket) -> None:
        session = await self._get_or_create(session_id)
        session.speak_connections.discard(websocket)
        await self._remove_if_empty(session)

    async def signal_barge_in(self, session_id: str) -> None:
        session = await self._get_or_create(session_id)
        await self._broadcast(session.speak_connections, {"type": "stop"})
        await self._broadcast_state(session, "idle")

    async def push_audio(self, session_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        session = await self._get_or_create(session_id)
        async with session.audio_lock:
            session.audio_chunks.append(bytes(chunk))
        await self._broadcast_state(session, "listening")

    async def reset_audio(self, session_id: str) -> None:
        session = await self._get_or_create(session_id)
        async with session.audio_lock:
            session.audio_chunks.clear()
        await self._broadcast_state(session, "idle")

    async def finalize_segment(self, session_id: str) -> None:
        session = await self._get_or_create(session_id)
        async with session.audio_lock:
            session.audio_chunks.clear()
        await self._broadcast_state(session, "idle")

    async def speak_text(
        self,
        session_id: str,
        text: str,
        *,
        emit_text: bool = False,
        context: str = "session_phase_4",
        phase: int = 4,
    ) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        session = await self._get_or_create(session_id)
        if emit_text:
            await self._broadcast(
                session.speak_connections,
                {
                    "type": "assistant_text",
                    "text": clean_text,
                    "isFinal": True,
                },
            )
        await self._broadcast_state(session, "speaking")
        try:
            audio_bytes = await asyncio.to_thread(
                _synthesize_mp3,
                clean_text,
                context,
                phase,
            )
            if not audio_bytes:
                raise RuntimeError("Leere TTS-Antwort.")
            message_id = f"assistant-{uuid.uuid4().hex}"
            chunk_size = 180_000
            total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
            for index in range(total_chunks):
                start = index * chunk_size
                end = min(len(audio_bytes), start + chunk_size)
                payload = {
                    "type": "audio",
                    "format": "audio/mpeg",
                    "data": base64.b64encode(audio_bytes[start:end]).decode("ascii"),
                    "seq": index,
                    "final": index == (total_chunks - 1),
                    "message_id": message_id,
                }
                if index == 0:
                    payload["text"] = clean_text
                await self._broadcast(session.speak_connections, payload)
            await self._broadcast(
                session.speak_connections,
                {"type": "end", "message_id": message_id},
            )
        except Exception as exc:
            await self._broadcast(
                session.speak_connections,
                {"type": "error", "message": f"TTS fehlgeschlagen: {exc}"},
            )
            raise
        finally:
            await self._broadcast_state(session, "idle")

    async def emit_phase4_start(self, session_id: str) -> ChatResponse:
        response = await asyncio.to_thread(_start_phase4_session, session_id)
        session = await self._get_or_create(session_id)
        await self._broadcast(session.speak_connections, self._assistant_payload(response))
        await self.speak_text(
            session_id,
            response.reply,
            emit_text=False,
            context="session_phase_4",
            phase=4,
        )
        return response

    async def handle_user_text(self, session_id: str, text: str) -> ChatResponse:
        clean_text = text.strip()
        if not clean_text:
            raise HTTPException(status_code=400, detail="message fehlt.")
        session = await self._get_or_create(session_id)
        async with session.processing_lock:
            await self._broadcast_state(session, "processing")
            transcript_payload = {
                "type": "transcript",
                "text": clean_text,
                "isFinal": True,
                "session_id": session_id,
            }
            await self._broadcast(session.listen_connections, transcript_payload)
            if _should_start_phase4_from_voice(session_id, clean_text):
                response = await asyncio.to_thread(_start_phase4_session, session_id)
            else:
                response = await asyncio.to_thread(
                    _chat_turn_internal,
                    session_id,
                    clean_text,
                )
            await self._broadcast(session.speak_connections, self._assistant_payload(response))
            await self.speak_text(
                session_id,
                response.reply,
                emit_text=False,
                context="session_phase_4",
                phase=4,
            )
            return response


voice_loop_manager = VoiceLoopManager()


def _resolve_ws_session_id(websocket: WebSocket) -> str:
    return (
        (websocket.query_params.get("session_id") or "").strip()
        or (websocket.query_params.get("token") or "").strip()
        or (websocket.query_params.get("user_id") or "").strip()
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "hypnose-chat-backend", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tts-audio")
def post_tts_audio(payload: TtsPayload):
    try:
        audio_bytes = _synthesize_mp3(
            payload.text.strip(),
            context=payload.context,
            phase=payload.phase,
        )
    except Exception as exc:
        logging.exception("Failed to synthesize TTS audio.")
        raise HTTPException(status_code=500, detail="TTS audio could not be generated.") from exc

    return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")


@app.post("/session/reset")
def reset_session(request: ResetSessionRequest) -> dict[str, str]:
    session_id = request.session_id.strip()
    with _history_lock:
        _session_history.pop(session_id, None)
        _phase4_state.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


@app.post("/phase4/start", response_model=ChatResponse)
def phase4_start(request: Phase4StartRequest) -> ChatResponse:
    return _start_phase4_session(request.session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _chat_turn_internal(
        request.session_id,
        request.message,
        reset_session=request.reset_session,
    )


@app.websocket("/audio/speak")
async def ws_audio_speak(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = _resolve_ws_session_id(websocket)
    if not session_id:
        await websocket.close(code=1008, reason="session_id fehlt")
        return
    await voice_loop_manager.register_speaker(session_id, websocket)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            if not text_data:
                continue
            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                await voice_loop_manager.handle_user_text(session_id, text_data)
                continue

            msg_type = (payload.get("type") or payload.get("typ") or "").strip().lower()
            if msg_type == "ping":
                continue
            if msg_type == "stop":
                await voice_loop_manager.signal_barge_in(session_id)
                continue
            if msg_type == "start_phase4":
                await voice_loop_manager.emit_phase4_start(session_id)
                continue
            if msg_type == "speak_text":
                speak_text = str(payload.get("text") or "").strip()
                if speak_text:
                    await voice_loop_manager.speak_text(
                        session_id,
                        speak_text,
                        emit_text=payload.get("emit_text", True) is True,
                    )
                continue
            if msg_type == "user_text":
                user_text = str(payload.get("text") or "").strip()
                if user_text:
                    await voice_loop_manager.handle_user_text(session_id, user_text)
                continue

            fallback_text = str(payload.get("text") or "").strip()
            if fallback_text:
                await voice_loop_manager.handle_user_text(session_id, fallback_text)
    except WebSocketDisconnect:
        pass
    finally:
        await voice_loop_manager.unregister_speaker(session_id, websocket)


@app.websocket("/audio/listen")
async def ws_audio_listen(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = _resolve_ws_session_id(websocket)
    if not session_id:
        await websocket.close(code=1008, reason="session_id fehlt")
        return
    await voice_loop_manager.register_listener(session_id, websocket)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk is not None:
                await voice_loop_manager.push_audio(session_id, chunk)
                continue

            text_data = message.get("text")
            if not text_data:
                continue

            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                await voice_loop_manager.handle_user_text(session_id, text_data)
                continue

            msg_type = (payload.get("type") or payload.get("typ") or "").strip().lower()
            if msg_type == "ping":
                continue
            if msg_type == "segment":
                await voice_loop_manager.finalize_segment(session_id)
                continue
            if msg_type == "reset":
                await voice_loop_manager.reset_audio(session_id)
                continue
            if msg_type == "barge_in":
                await voice_loop_manager.signal_barge_in(session_id)
                continue
            if msg_type == "user_text":
                user_text = str(payload.get("text") or "").strip()
                if user_text:
                    await voice_loop_manager.handle_user_text(session_id, user_text)
                continue

            fallback_text = str(payload.get("text") or "").strip()
            if fallback_text:
                await voice_loop_manager.handle_user_text(session_id, fallback_text)
    except WebSocketDisconnect:
        pass
    finally:
        await voice_loop_manager.unregister_listener(session_id, websocket)

