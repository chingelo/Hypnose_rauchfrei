import copy
import json
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TTS_SETTINGS_PATH = BASE_DIR / "db" / "tts_settings.json"

DEFAULT_TTS_SETTINGS = {
    "profiles": {
        "default": {
            "speaking_rate": 0.90,
            "lead_break_ms": 240,
            "trail_break_ms": 260,
            "newline_break_ms": 280,
            "paragraph_break_ms": 720,
            "sentence_break_ms": 150,
            "comma_break_ms": 45,
            "ellipsis_break_ms": 300,
        },
        "assistant_short": {
            "speaking_rate": 0.82,
            "lead_break_ms": 220,
            "trail_break_ms": 240,
            "newline_break_ms": 260,
            "paragraph_break_ms": 620,
            "sentence_break_ms": 170,
            "comma_break_ms": 55,
            "ellipsis_break_ms": 320,
        },
        "questionnaire": {
            "speaking_rate": 1.00,
            "lead_break_ms": 240,
            "trail_break_ms": 200,
            "newline_break_ms": 220,
            "paragraph_break_ms": 560,
            "sentence_break_ms": 120,
            "comma_break_ms": 30,
            "ellipsis_break_ms": 220,
        },
        "emdr": {
            "speaking_rate": 0.84,
            "lead_break_ms": 260,
            "trail_break_ms": 300,
            "newline_break_ms": 320,
            "paragraph_break_ms": 900,
            "sentence_break_ms": 180,
            "comma_break_ms": 50,
            "ellipsis_break_ms": 360,
        },
        "session_default": {
            "speaking_rate": 0.76,
            "lead_break_ms": 320,
            "trail_break_ms": 360,
            "newline_break_ms": 360,
            "paragraph_break_ms": 940,
            "sentence_break_ms": 220,
            "comma_break_ms": 70,
            "ellipsis_break_ms": 420,
        },
        "session_phase_1": {"speaking_rate": 0.82},
        "session_phase_2": {"speaking_rate": 0.82},
        "session_phase_3": {"speaking_rate": 0.78},
        "session_phase_4": {"speaking_rate": 0.74},
        "session_phase_5": {"speaking_rate": 0.74},
        "session_phase_6": {"speaking_rate": 0.78},
    }
}

PROFILE_ALIASES = {
    "fragebogen": "questionnaire",
    "questionnaire": "questionnaire",
    "guided_questionnaire": "questionnaire",
    "emdr": "emdr",
    "session": "session_default",
    "hypnose": "session_default",
    "phase": "session_default",
    "assistant_short": "assistant_short",
    "assistant": "assistant_short",
    "system_short": "assistant_short",
    "default": "default",
}

_LOCK = threading.Lock()


def _to_float(value, fallback):
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return float(fallback)


def _to_int(value, fallback):
    try:
        return int(float(str(value).strip().replace(",", ".")))
    except Exception:
        return int(fallback)


def _clamp(value, low, high):
    return max(low, min(value, high))


def _normalize_profile(raw_profile, fallback_profile):
    fallback = fallback_profile or {}
    raw = raw_profile or {}
    return {
        "speaking_rate": _clamp(_to_float(raw.get("speaking_rate"), fallback.get("speaking_rate", 0.9)), 0.55, 1.35),
        "lead_break_ms": _clamp(_to_int(raw.get("lead_break_ms"), fallback.get("lead_break_ms", 220)), 0, 5000),
        "trail_break_ms": _clamp(_to_int(raw.get("trail_break_ms"), fallback.get("trail_break_ms", 260)), 0, 5000),
        "newline_break_ms": _clamp(_to_int(raw.get("newline_break_ms"), fallback.get("newline_break_ms", 280)), 0, 5000),
        "paragraph_break_ms": _clamp(
            _to_int(raw.get("paragraph_break_ms"), fallback.get("paragraph_break_ms", 720)),
            0,
            10000,
        ),
        "sentence_break_ms": _clamp(
            _to_int(raw.get("sentence_break_ms"), fallback.get("sentence_break_ms", 150)),
            0,
            3000,
        ),
        "comma_break_ms": _clamp(
            _to_int(raw.get("comma_break_ms"), fallback.get("comma_break_ms", 45)),
            0,
            1500,
        ),
        "ellipsis_break_ms": _clamp(
            _to_int(raw.get("ellipsis_break_ms"), fallback.get("ellipsis_break_ms", 300)),
            0,
            5000,
        ),
    }


def _build_effective_settings(raw_settings):
    defaults = copy.deepcopy(DEFAULT_TTS_SETTINGS)
    raw_profiles = (raw_settings or {}).get("profiles")
    if not isinstance(raw_profiles, dict):
        return defaults

    profiles = defaults["profiles"]
    for key, value in raw_profiles.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        key_norm = key.strip().lower()
        if not key_norm:
            continue
        fallback = profiles.get(key_norm) or profiles.get("default") or {}
        merged = copy.deepcopy(fallback)
        merged.update(value)
        profiles[key_norm] = _normalize_profile(merged, fallback)
    return defaults


def _read_settings_file():
    if not TTS_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(TTS_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_tts_settings():
    with _LOCK:
        return _build_effective_settings(_read_settings_file())


def save_tts_settings(payload):
    with _LOCK:
        normalized = _build_effective_settings(payload)
        TTS_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TTS_SETTINGS_PATH.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized


def _resolve_profile_name(context=None, phase=None, settings=None):
    config = settings or load_tts_settings()
    profiles = config.get("profiles", {})

    context_norm = str(context or "default").strip().lower()
    profile_name = PROFILE_ALIASES.get(context_norm, context_norm)

    if context_norm.startswith("session_phase_"):
        if context_norm in profiles:
            return context_norm
        return "session_default" if "session_default" in profiles else "default"

    if profile_name in {"session_default", "session", "hypnose", "phase"}:
        if phase is not None:
            try:
                phase_num = int(phase)
                phase_key = f"session_phase_{phase_num}"
                if phase_key in profiles:
                    return phase_key
            except Exception:
                pass
        return "session_default" if "session_default" in profiles else "default"

    if profile_name in profiles:
        return profile_name

    return "default"


def get_tts_profile(context=None, phase=None, settings=None):
    config = settings or load_tts_settings()
    profiles = config.get("profiles", {})
    name = _resolve_profile_name(context=context, phase=phase, settings=config)
    profile = profiles.get(name) or profiles.get("default") or _normalize_profile({}, {})
    return name, profile
