from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_gold_finetune_data import _render_question, _runtime_slots_for
from build_finetune_splits import _split_groups, _write_jsonl
from run_session_sandbox import (
    _classify_focus_reference,
    _display_trigger_focus_ref,
    _extract_named_person_label,
)
from session_sandbox_orchestrator import SemanticNodeSpec, get_node_spec


ROOT_DIR = Path(__file__).resolve().parent
V1_DIR = ROOT_DIR / "finetune_data"
V2_DIR = V1_DIR / "v2"


ROUTING_EDGE_CASES: list[dict[str, Any]] = [
    {
        "label": "dark_feeling_typo_pressure",
        "node_id": "dark_scene_feeling_intensity",
        "user_reply": "ich spuehre ein druck",
        "intent": "feeling_and_intensity",
    },
    {
        "label": "dark_feeling_scale_only",
        "node_id": "dark_scene_feeling_intensity",
        "user_reply": "acht von zehn",
        "intent": "intensity_only",
    },
    {
        "label": "dark_immediate_pressure_phrase",
        "node_id": "dark_scene_immediate_feeling",
        "user_reply": "druck in der brust",
        "intent": "ready",
    },
    {
        "label": "dark_perception_smell",
        "node_id": "dark_scene_perception",
        "user_reply": "ich rieche rauch",
        "intent": "other_sense",
    },
    {
        "label": "dark_perception_audio_colloquial",
        "node_id": "dark_scene_perception",
        "user_reply": "ich hoer wie die lachen",
        "intent": "audio",
    },
    {
        "label": "dark_perception_visual_person",
        "node_id": "dark_scene_perception",
        "user_reply": "ich sehe meinen vater auf dem schulhof",
        "intent": "visual",
    },
    {
        "label": "dark_mode_smell_only",
        "node_id": "dark_scene_mode_clarify",
        "user_reply": "ich rieche nur rauch",
        "intent": "other_sense",
    },
    {
        "label": "dark_mode_audio_only",
        "node_id": "dark_scene_mode_clarify",
        "user_reply": "ich hoer nur lachen",
        "intent": "audio",
    },
    {
        "label": "dark_mode_nothing_colloquial",
        "node_id": "dark_scene_mode_clarify",
        "user_reply": "gar nichts",
        "intent": "nothing",
    },
    {
        "label": "scene_access_smell",
        "node_id": "scene_access_followup",
        "user_reply": "ich rieche rauch",
        "intent": "nonvisual_access",
    },
    {
        "label": "scene_access_visual_dark",
        "node_id": "scene_access_followup",
        "user_reply": "ich sehe eine dunkle ecke",
        "intent": "visual_dark",
    },
    {
        "label": "origin_owner_external_colloquial",
        "node_id": "origin_cause_owner",
        "user_reply": "eher von denen",
        "intent": "someone_else",
        "runtime_slots": {"trigger_focus_ref": "die freunde auf dem pausenhof"},
    },
    {
        "label": "origin_owner_internal_colloquial",
        "node_id": "origin_cause_owner",
        "user_reply": "eher aus mir selbst",
        "intent": "self",
        "runtime_slots": {"trigger_focus_ref": "die freunde auf dem pausenhof"},
    },
    {
        "label": "origin_kind_person_brother",
        "node_id": "origin_other_target_kind",
        "user_reply": "mein bruder",
        "intent": "person",
        "runtime_slots": {"trigger_focus_ref": "mein bruder"},
    },
    {
        "label": "origin_kind_group_smoking_clique",
        "node_id": "origin_other_target_kind",
        "user_reply": "die raucher clique",
        "intent": "group",
        "runtime_slots": {"trigger_focus_ref": "die raucher clique"},
    },
    {
        "label": "origin_kind_other_smoke_smell",
        "node_id": "origin_other_target_kind",
        "user_reply": "der geruch von rauch",
        "intent": "other",
        "runtime_slots": {"trigger_focus_ref": "der geruch von rauch"},
    },
    {
        "label": "origin_kind_other_color",
        "node_id": "origin_other_target_kind",
        "user_reply": "die farbe rot",
        "intent": "other",
        "runtime_slots": {"trigger_focus_ref": "die farbe rot"},
    },
    {
        "label": "group_source_whole_colloquial",
        "node_id": "group_source_kind",
        "user_reply": "alle zusammen",
        "intent": "whole_group",
    },
    {
        "label": "group_source_one_person_colloquial",
        "node_id": "group_source_kind",
        "user_reply": "eher peter selbst",
        "intent": "one_person",
    },
    {
        "label": "group_source_multiple_colloquial",
        "node_id": "group_source_kind",
        "user_reply": "mehrere einzelne daraus",
        "intent": "multiple_people",
    },
    {
        "label": "group_scope_representative_colloquial",
        "node_id": "group_whole_scope",
        "user_reply": "eine person reicht stellvertretend",
        "intent": "representative_enough",
    },
    {
        "label": "group_scope_multiple_colloquial",
        "node_id": "group_whole_scope",
        "user_reply": "wir muessen mehrere einzeln anschauen",
        "intent": "multiple_required",
    },
    {
        "label": "group_core_real_explanation",
        "node_id": "group_person_trigger_core",
        "user_reply": "ja sie lachen weil ich der einzige bin der nicht raucht",
        "intent": "ready",
    },
    {
        "label": "group_core_incomplete_but_meaningful",
        "node_id": "group_person_trigger_core",
        "user_reply": "noch nicht ganz klar, aber es hat mit ausgrenzung zu tun",
        "intent": "ready",
    },
    {
        "label": "person_switch_why_belonging",
        "node_id": "person_switch_why",
        "user_reply": "weil er dazugehoeren wollte",
        "intent": "ready",
    },
]


SLOT_EDGE_CASES: dict[str, list[dict[str, Any]]] = {
    "focus_kind": [
        {"user_reply": "mein bruder", "expected": {"focus_kind": "person"}},
        {"user_reply": "meine oma", "expected": {"focus_kind": "person"}},
        {"user_reply": "frau meier", "expected": {"focus_kind": "person"}},
        {"user_reply": "der typ mit der zigarette", "expected": {"focus_kind": "person"}},
        {"user_reply": "die anderen da hinten", "expected": {"focus_kind": "group"}},
        {"user_reply": "ein haufen leute", "expected": {"focus_kind": "group"}},
        {"user_reply": "die aus der raucherecke", "expected": {"focus_kind": "group"}},
        {"user_reply": "alle am fenster", "expected": {"focus_kind": "group"}},
        {"user_reply": "der gestank", "expected": {"focus_kind": "other"}},
        {"user_reply": "das rot", "expected": {"focus_kind": "other"}},
        {"user_reply": "die spannung im raum", "expected": {"focus_kind": "other"}},
        {"user_reply": "rauchgeruch", "expected": {"focus_kind": "other"}},
        {"user_reply": "ein komisches gefuehl", "expected": {"focus_kind": "other"}},
        {"user_reply": "die farbe blau", "expected": {"focus_kind": "other"}},
    ],
    "named_person_label": [
        {"user_reply": "mein bruder", "expected": {"named_person": "Mein Bruder"}},
        {"user_reply": "meine oma", "expected": {"named_person": "Meine Oma"}},
        {"user_reply": "frau meier", "expected": {"named_person": "Frau Meier"}},
        {"user_reply": "die anderen da hinten", "expected": {"named_person": None}},
        {"user_reply": "rauchgeruch", "expected": {"named_person": None}},
        {"user_reply": "die farbe blau", "expected": {"named_person": None}},
    ],
    "display_trigger_focus_ref": [
        {"trigger_focus_ref": "die anderen da hinten", "expected": {"display_trigger_focus_ref": "diese Gruppe"}},
        {"trigger_focus_ref": "ein haufen leute", "expected": {"display_trigger_focus_ref": "diese Gruppe"}},
        {"trigger_focus_ref": "die aus der raucherecke", "expected": {"display_trigger_focus_ref": "diese Gruppe"}},
        {"trigger_focus_ref": "der gestank", "expected": {"display_trigger_focus_ref": "der gestank"}},
        {"trigger_focus_ref": "die farbe blau", "expected": {"display_trigger_focus_ref": "die farbe blau"}},
    ],
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(row: dict[str, Any]) -> str:
        return hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()

    return sorted(rows, key=_key)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def _dedupe_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        fingerprint = json.dumps(
            {
                "task": row["task"],
                "input": _canonicalize(row["input"]),
                "output": _canonicalize(row["output"]),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        deduped.append(row)
    return deduped, duplicates


def _routing_edge_record(index: int, case: dict[str, Any]) -> dict[str, Any]:
    node_id = str(case["node_id"])
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise TypeError(f"{node_id} is not semantic")
    intent = str(case["intent"])
    rule = spec.routing_rules[intent]
    runtime_slots = _runtime_slots_for(node_id)
    runtime_slots.update(case.get("runtime_slots", {}))
    return {
        "id": f"routing-v2-{index:04d}",
        "task": "routing",
        "source": ["gold_v2_edge_cases", "bug_history"],
        "input": {
            "node_id": node_id,
            "question_text": _render_question(node_id, runtime_slots),
            "runtime_slots": runtime_slots,
            "session_hint": "",
            "user_reply": case["user_reply"],
        },
        "output": {
            "intent": intent,
            "action": rule["action"],
            "next_node": rule["next_node"],
        },
        "label": case["label"],
    }


def build_routing_v2() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = _load_jsonl(V1_DIR / "routing_gold.jsonl")
    edge_rows = [_routing_edge_record(index + 1, case) for index, case in enumerate(ROUTING_EDGE_CASES)]
    combined = base_rows + edge_rows
    deduped, duplicates_removed = _dedupe_records(combined)

    synthetic_markers = {
        "placeholder_banane": 0,
        "generic_repeat": 0,
        "generic_question": 0,
        "generic_support": 0,
        "generic_abort": 0,
    }
    for row in base_rows:
        reply = str(row["input"]["user_reply"]).strip().lower()
        if reply == "banane":
            synthetic_markers["placeholder_banane"] += 1
        elif reply == "wiederhole die frage bitte":
            synthetic_markers["generic_repeat"] += 1
        elif reply == "wie meinst du das?":
            synthetic_markers["generic_question"] += 1
        elif reply == "es ist mir gerade zu viel":
            synthetic_markers["generic_support"] += 1
        elif reply == "ich will abbrechen":
            synthetic_markers["generic_abort"] += 1

    review = {
        "v1_count": len(base_rows),
        "edge_case_count_added": len(edge_rows),
        "duplicates_removed": duplicates_removed,
        "v2_count": len(deduped),
        "synthetic_marker_counts_v1": synthetic_markers,
        "edge_labels": [case["label"] for case in ROUTING_EDGE_CASES],
        "top_node_counts_v2": Counter(row["input"]["node_id"] for row in deduped).most_common(15),
    }
    return deduped, review


def _slot_edge_record(index: int, task: str, case: dict[str, Any]) -> dict[str, Any]:
    input_payload = {"user_reply": case["user_reply"]} if "user_reply" in case else {"trigger_focus_ref": case["trigger_focus_ref"]}
    return {
        "id": f"slot-v2-{index:04d}",
        "task": task,
        "source": ["gold_v2_edge_cases", "bug_history"],
        "input": input_payload,
        "output": case["expected"],
    }


def build_slot_v2() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = _load_jsonl(V1_DIR / "slot_extraction_gold.jsonl")
    edge_rows: list[dict[str, Any]] = []
    idx = 0
    helper_alignment: list[dict[str, Any]] = []

    for task, cases in SLOT_EDGE_CASES.items():
        for case in cases:
            idx += 1
            edge_rows.append(_slot_edge_record(idx, task, case))
            if task == "focus_kind":
                actual = {"focus_kind": _classify_focus_reference(case["user_reply"])}
            elif task == "named_person_label":
                actual = {"named_person": _extract_named_person_label(case["user_reply"])}
            else:
                actual = {"display_trigger_focus_ref": _display_trigger_focus_ref(case["trigger_focus_ref"])}
            helper_alignment.append(
                {
                    "task": task,
                    "input": case.get("user_reply") or case.get("trigger_focus_ref"),
                    "expected": case["expected"],
                    "current_helper_output": actual,
                    "matches_current_helper": actual == case["expected"],
                }
            )

    combined = base_rows + edge_rows
    deduped, duplicates_removed = _dedupe_records(combined)
    mismatches = [row for row in helper_alignment if not row["matches_current_helper"]]
    review = {
        "v1_count": len(base_rows),
        "edge_case_count_added": len(edge_rows),
        "duplicates_removed": duplicates_removed,
        "v2_count": len(deduped),
        "task_counts_v2": Counter(row["task"] for row in deduped),
        "helper_alignment_total": len(helper_alignment),
        "helper_alignment_mismatches": len(mismatches),
        "helper_alignment_examples": mismatches[:20],
    }
    return deduped, review


def build_v2_splits(rows: list[dict[str, Any]], *, dataset: str) -> dict[str, list[dict[str, Any]]]:
    if dataset == "routing":
        train_rows, eval_rows = _split_groups(
            _stable_order(rows),
            group_key=lambda row: str(row["input"]["node_id"]),
            eval_ratio=0.2,
        )
    elif dataset == "slot_extraction":
        train_rows, eval_rows = _split_groups(
            _stable_order(rows),
            group_key=lambda row: str(row["task"]),
            eval_ratio=0.2,
        )
    else:
        raise ValueError(dataset)
    return {"train": train_rows, "eval": eval_rows}


def write_v2_artifacts() -> dict[str, Any]:
    V2_DIR.mkdir(parents=True, exist_ok=True)

    routing_rows, routing_review = build_routing_v2()
    slot_rows, slot_review = build_slot_v2()

    _write_jsonl(V2_DIR / "routing_gold_v2.jsonl", routing_rows)
    _write_jsonl(V2_DIR / "slot_extraction_gold_v2.jsonl", slot_rows)

    routing_split = build_v2_splits(routing_rows, dataset="routing")
    slot_split = build_v2_splits(slot_rows, dataset="slot_extraction")
    _write_jsonl(V2_DIR / "routing_train_v2.jsonl", routing_split["train"])
    _write_jsonl(V2_DIR / "routing_eval_v2.jsonl", routing_split["eval"])
    _write_jsonl(V2_DIR / "slot_extraction_train_v2.jsonl", slot_split["train"])
    _write_jsonl(V2_DIR / "slot_extraction_eval_v2.jsonl", slot_split["eval"])

    review = {
        "version": 2,
        "datasets": {
            "routing": routing_review,
            "slot_extraction": slot_review,
        },
        "split_counts": {
            "routing_train_v2.jsonl": len(routing_split["train"]),
            "routing_eval_v2.jsonl": len(routing_split["eval"]),
            "slot_extraction_train_v2.jsonl": len(slot_split["train"]),
            "slot_extraction_eval_v2.jsonl": len(slot_split["eval"]),
        },
    }
    (V2_DIR / "gold_v2_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (V2_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Gold V2",
                "",
                "Gold V2 haertet zunaechst nur `routing` und `slot_extraction`.",
                "",
                "Ziele:",
                "- reale Problemfaelle aus der Bug-Historie aufnehmen",
                "- offensichtliche Dubletten entfernen",
                "- V1-Synthese beibehalten, aber durch Grenzfaelle ergaenzen",
                "- sichtbar machen, wo die aktuelle Helper-Logik semantisch noch zu grob ist",
                "",
                "Dateien:",
                "- `routing_gold_v2.jsonl`",
                "- `routing_train_v2.jsonl`",
                "- `routing_eval_v2.jsonl`",
                "- `slot_extraction_gold_v2.jsonl`",
                "- `slot_extraction_train_v2.jsonl`",
                "- `slot_extraction_eval_v2.jsonl`",
                "- `gold_v2_review.json`",
                "",
                "Generator:",
                "- `backend/build_gold_v2_datasets.py`",
                "",
                "Aktueller Stand:",
                "- Gold V2 dient als Soll-Zustand fuer Runtime-Heuristik und spaetere Modell-Evals",
                "- `gold_v2_review.json` zeigt, ob die aktuelle Helper-Logik mit diesem Soll bereits uebereinstimmt",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return review


def main() -> None:
    review = write_v2_artifacts()
    print(json.dumps(review["split_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
