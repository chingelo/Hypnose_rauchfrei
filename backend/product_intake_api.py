from __future__ import annotations

import copy
import json
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from session_access_integration import seed_session_status_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = PROJECT_ROOT / "backend" / "content_reference" / "catalog"
RAUCHFREI_ASSET_BASE = "/assets/rauchfrei"

product_router = APIRouter(tags=["product"])


RAUCHFREI_PROGRAM = {
    "key": "rauchfrei",
    "name": "Rauchfrei-Paket",
    "description": (
        "Gefuehrter Rauchfrei-Ablauf mit Vorbereitung, Hauptsession und Stabilisierung."
    ),
    "steps": [
        {
            "phase": 1,
            "title": "Start und Orientierung",
            "asset": "02_Begleitunterlagen/01_Dein_Weg_zur_Rauchfreiheit.pdf",
            "instruction": "Schaffe dir einen ruhigen Rahmen und richte dich bewusst auf den Neustart aus.",
        },
        {
            "phase": 2,
            "title": "Fragebogen ausfuellen",
            "asset": "03_Fragebogen/Fragebogen_Rauchstopp.docx",
            "instruction": "Klaere Motivation, Trigger und Zielbild so konkret wie moeglich.",
        },
        {
            "phase": 3,
            "title": "Selbstreflexion",
            "asset": "02_Begleitunterlagen/03_Rauchfreitagebuch.pdf",
            "instruction": "Arbeite Trigger, Vorteile und dein Nichtraucher-Zielbild klar heraus.",
        },
        {
            "phase": 4,
            "title": "Vorbereitung",
            "asset": "01_Audio/02_Vorbereitung.mp3",
            "instruction": "Hoere die Vorbereitung in einem sicheren und ungestoerten Rahmen.",
        },
        {
            "phase": 5,
            "title": "Hauptsession",
            "asset": "01_Audio/01_Hauptsession_Rauchfrei.mp3",
            "instruction": "Hoere die Hauptsession vollstaendig und ohne Ablenkung durch.",
        },
        {
            "phase": 6,
            "title": "Stabilisierung",
            "asset": "01_Audio/03_Entspannung_fuer_den_Alltag.mp3",
            "instruction": "Nutze die Stabilisierung, um den neuen Zustand im Alltag zu festigen.",
        },
        {
            "phase": 7,
            "title": "Workbook begleiten",
            "asset": "02_Begleitunterlagen/02_Workbook_Rauchfreiheit.pdf",
            "instruction": "Arbeite parallel mit dem Workbook und halte Erkenntnisse schriftlich fest.",
        },
    ],
}

DEFAULT_RAUCHFREI_GUIDED_FORMS = {
    "questionnaire": {
        "title": "Gefuehrter Rauchstopp-Fragebogen",
        "questions": [
            "Seit wie vielen Jahren rauchst du schon?",
            "Wie viele Zigaretten rauchst du pro Tag?",
            "In welchen Situationen greifst du am haeufigsten zur Zigarette?",
            "Was stoert dich am Rauchen aktuell am meisten?",
            "Was waere der groesste Vorteil fuer dich, wenn du rauchfrei bist?",
            "Was hat bei frueheren Aufhoerversuchen funktioniert und was nicht?",
            "Was ist dein klares Zielbild als Nichtraucher in den naechsten 30 Tagen?",
        ],
    },
    "journal": {
        "title": "Gefuehrtes Rauchfrei-Tagebuch",
        "questions": [
            "Wann war heute dein staerkster Rauchimpuls?",
            "Was hast du in diesem Moment gefuehlt oder gedacht?",
            "Wie stark war der Impuls auf einer Skala von 1 bis 10?",
            "Was hat dir geholfen, rauchfrei zu bleiben oder wieder in deine Entscheidung zu finden?",
            "Worauf bist du heute stolz in deinem Rauchfrei-Prozess?",
        ],
    },
}


class IntakePayload(BaseModel):
    vorname: str = Field(..., min_length=1)
    email: str | None = None
    anliegen: str = Field(..., min_length=3)
    datenschutz_zustimmung: bool
    symptom_key: str | None = None
    technik_key: str | None = None
    program_type: str = "standard"


class GuidedAnswersPayload(BaseModel):
    form_key: str
    answers: list[str] = Field(default_factory=list)
    session_id: str | None = None


class GuidedFollowupPayload(BaseModel):
    form_key: str
    question_index: int = 0
    answer: str = ""


def _safe_json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return copy.deepcopy(default)
    except Exception:
        return copy.deepcopy(default)


def _safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _normalize_program_type(value: str | None) -> str:
    return "rauchfrei" if (value or "").strip().lower() == "rauchfrei" else "standard"


def _runtime_state_root() -> Path:
    configured = (os.getenv("TEST_APP_RUNTIME_STATE_DIR") or "").strip()
    if configured:
        return Path(configured).resolve()
    return (PROJECT_ROOT / "backend" / "runtime_state").resolve()


def _guided_entries_dir() -> Path:
    return _runtime_state_root() / "guided_entries"


def _intake_records_path() -> Path:
    return _runtime_state_root() / "intake_records.json"


def _rauchfrei_package_root() -> Path | None:
    configured = (os.getenv("RAUCHFREI_PACKAGE_ROOT") or "").strip()
    if not configured:
        return None
    path = Path(configured).resolve()
    return path if path.is_dir() else None


def register_rauchfrei_assets(app: FastAPI) -> None:
    package_root = _rauchfrei_package_root()
    if package_root is None:
        return
    mount_exists = any(getattr(route, "path", "") == RAUCHFREI_ASSET_BASE for route in app.routes)
    if not mount_exists:
        app.mount(
            RAUCHFREI_ASSET_BASE,
            StaticFiles(directory=str(package_root)),
            name="rauchfrei-assets",
        )


def _build_rauchfrei_program_payload() -> dict[str, Any]:
    package_root = _rauchfrei_package_root()
    steps: list[dict[str, Any]] = []
    for step in RAUCHFREI_PROGRAM["steps"]:
        asset_relative = str(step["asset"]).replace("\\", "/")
        asset_exists = False
        if package_root is not None:
            asset_exists = (package_root / Path(asset_relative)).exists()
        steps.append(
            {
                **step,
                "asset_relative": asset_relative,
                "asset_exists": asset_exists,
                "asset_url": (
                    f"{RAUCHFREI_ASSET_BASE}/{asset_relative}"
                    if package_root is not None
                    else None
                ),
            }
        )
    return {
        **RAUCHFREI_PROGRAM,
        "assets_available": package_root is not None,
        "package_root": str(package_root) if package_root is not None else None,
        "steps": steps,
    }


def _guided_forms_path() -> Path:
    return CONTENT_ROOT / "forms" / "guided_forms.json"


def _normalize_guided_forms(payload: Any) -> dict[str, dict[str, Any]]:
    normalized = copy.deepcopy(DEFAULT_RAUCHFREI_GUIDED_FORMS)
    if not isinstance(payload, dict):
        return normalized
    for form_key, form_payload in payload.items():
        if not isinstance(form_payload, dict):
            continue
        title = str(form_payload.get("title") or "").strip()
        questions_raw = form_payload.get("questions")
        questions = []
        if isinstance(questions_raw, list):
            questions = [str(item).strip() for item in questions_raw if str(item).strip()]
        if not title or not questions:
            continue
        normalized[str(form_key).strip()] = {
            "title": title,
            "questions": questions,
        }
    return normalized


def _load_guided_forms() -> dict[str, dict[str, Any]]:
    return _normalize_guided_forms(_safe_json_load(_guided_forms_path(), {}))


def _store_guided_entry(session_id: str | None, form_key: str, items: list[dict[str, str]], summary_text: str) -> dict[str, str]:
    guided_entries_dir = _guided_entries_dir()
    guided_entries_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "form_key": form_key,
        "items": items,
        "summary_text": summary_text,
    }
    suffix = secrets.token_urlsafe(8)
    target = guided_entries_dir / f"{form_key}_{session_id or 'no_session'}_{suffix}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"storage": "file", "path": str(target)}


def _build_followup(form_key: str, question_index: int, answer: str) -> str:
    _ = question_index
    text = str(answer or "").strip()
    if not text:
        return "Magst du dazu noch etwas mehr sagen, damit das Muster klarer wird?"

    compact = text.lower()
    if any(token in compact for token in ("stress", "druck", "nerv", "ueberfordert", "überfordert")):
        return "Woran merkst du diesen Stress im Koerper am staerksten?"
    if any(token in compact for token in ("morgen", "kaffee", "auto", "pause", "abend")):
        return "Was koennte genau in diesem Moment dein neuer rauchfreier Ersatz sein?"
    if any(token in compact for token in ("angst", "unsicher", "allein", "traurig")):
        return "Was wuerdest du in diesem Moment stattdessen brauchen, um dich sicherer zu fuehlen?"
    if form_key == "journal":
        return "Was sagt dir diese Situation ueber deinen wichtigsten Trigger oder dein staerkstes Beduerfnis?"
    return "Wenn du noch genauer hinschaust: Was ist der eigentliche Kern hinter dieser Antwort?"


def _build_welcome_text(vorname: str) -> str:
    clean_name = str(vorname or "").strip()
    if clean_name:
        return f"Willkommen {clean_name}. Wir starten jetzt Schritt fuer Schritt in deinen Rauchfrei-Prozess."
    return "Willkommen. Wir starten jetzt Schritt fuer Schritt in deinen Rauchfrei-Prozess."


def _infer_problem_type(anliegen: str, program_type: str) -> str:
    if program_type == "rauchfrei":
        return "rauchfrei"
    compact = anliegen.lower()
    if any(token in compact for token in ("stress", "druck", "ueberfordert", "überfordert")):
        return "stress_regulation"
    if any(token in compact for token in ("angst", "panik", "unsicher")):
        return "anxiety_regulation"
    return "general"


def _infer_technik_key(payload: IntakePayload, problem_type: str) -> str | None:
    if payload.technik_key:
        return payload.technik_key.strip()
    if problem_type == "rauchfrei":
        return "rauchfrei_guided_session"
    if problem_type == "stress_regulation":
        return "regulation_support"
    if problem_type == "anxiety_regulation":
        return "safety_regulation"
    return None


def _load_intake_records() -> dict[str, dict[str, Any]]:
    payload = _safe_json_load(_intake_records_path(), {})
    if isinstance(payload, dict):
        return payload
    return {}


def _save_intake_records(records: dict[str, dict[str, Any]]) -> None:
    _safe_json_write(_intake_records_path(), records)


def _create_intake(payload: IntakePayload) -> dict[str, Any]:
    if not payload.datenschutz_zustimmung:
        raise HTTPException(status_code=400, detail="Datenschutz consent required.")

    normalized_program_type = _normalize_program_type(payload.program_type)
    problem_type = _infer_problem_type(payload.anliegen, normalized_program_type)
    technik_key = _infer_technik_key(payload, problem_type)
    selection_reason = (
        "rauchfrei default flow"
        if problem_type == "rauchfrei"
        else "local heuristic selection"
    )

    anamnese_id = f"anamnese_{secrets.token_urlsafe(8)}"
    client_id = f"client_{secrets.token_urlsafe(8)}"
    session_id = f"session_{secrets.token_urlsafe(10)}"
    welcome_text = _build_welcome_text(payload.vorname)

    record = {
        "anamnese_id": anamnese_id,
        "client_id": client_id,
        "session_id": session_id,
        "vorname": payload.vorname.strip(),
        "email": (payload.email or "").strip() or None,
        "anliegen": payload.anliegen.strip(),
        "symptom_key": payload.symptom_key,
        "technik_key": technik_key,
        "problem_type": problem_type,
        "technik_selection_reason": selection_reason,
        "program_type": normalized_program_type,
        "welcome_text": welcome_text,
        "status": "created",
    }

    records = _load_intake_records()
    records[session_id] = record
    _save_intake_records(records)

    seed_session_status_record(
        session_id=session_id,
        status="ready",
        program_type=normalized_program_type,
        source="intake",
        last_step=1,
        extra_fields={
            "welcome_text": welcome_text,
            "anamnese_id": anamnese_id,
            "client_id": client_id,
            "problem_type": problem_type,
            "technik_key": technik_key,
            "technik_selection_reason": selection_reason,
            "vorname": payload.vorname.strip(),
            "email": (payload.email or "").strip() or None,
            "symptom_key": payload.symptom_key,
        },
    )

    return {
        "status": "created",
        "anamnese_id": anamnese_id,
        "client_id": client_id,
        "session_id": session_id,
        "welcome_text": welcome_text,
        "technik_key": technik_key,
        "problem_type": problem_type,
        "technik_primary_candidate": technik_key,
        "technik_secondary_candidates": [],
        "technik_selection_reason": selection_reason,
        "program_type": normalized_program_type,
    }


@product_router.get("/api/rauchfrei-package")
def get_rauchfrei_package() -> dict[str, Any]:
    return _build_rauchfrei_program_payload()


@product_router.get("/api/rauchfrei-guided-forms")
def get_rauchfrei_guided_forms() -> dict[str, dict[str, Any]]:
    return _load_guided_forms()


@product_router.post("/api/rauchfrei-guided-summary")
def post_rauchfrei_guided_summary(payload: GuidedAnswersPayload) -> dict[str, Any]:
    forms = _load_guided_forms()
    form = forms.get(payload.form_key)
    if not form:
        raise HTTPException(status_code=404, detail="Unknown form.")

    pairs = []
    for question, answer in zip(form["questions"], payload.answers):
        answer_text = str(answer or "").strip() or "[Keine Antwort]"
        pairs.append({"question": question, "answer": answer_text})

    summary_lines = [form["title"]]
    for index, pair in enumerate(pairs, start=1):
        summary_lines.append(f"{index}. {pair['question']}")
        summary_lines.append(f"Antwort: {pair['answer']}")
    summary_text = "\n".join(summary_lines)
    storage_info = _store_guided_entry(payload.session_id, payload.form_key, pairs, summary_text)

    return {
        "form_key": payload.form_key,
        "title": form["title"],
        "items": pairs,
        "summary_text": summary_text,
        "storage": storage_info,
    }


@product_router.post("/api/rauchfrei-guided-followup")
def post_rauchfrei_guided_followup(payload: GuidedFollowupPayload) -> dict[str, Any]:
    forms = _load_guided_forms()
    form = forms.get(payload.form_key)
    if not form:
        raise HTTPException(status_code=404, detail="Unknown form.")
    if payload.question_index < 0 or payload.question_index >= len(form["questions"]):
        raise HTTPException(status_code=400, detail="Invalid question index.")

    return {
        "form_key": payload.form_key,
        "question_index": payload.question_index,
        "followup": _build_followup(payload.form_key, payload.question_index, payload.answer),
    }


@product_router.post("/api/intake")
def intake(payload: IntakePayload) -> dict[str, Any]:
    return _create_intake(payload)
