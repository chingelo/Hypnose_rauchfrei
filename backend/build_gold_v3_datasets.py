from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_finetune_splits import _split_groups, _write_jsonl
from build_gold_v2_datasets import (
    SLOT_EDGE_CASES,
    ROUTING_EDGE_CASES,
    _dedupe_records,
    _routing_edge_record,
    _slot_edge_record,
    _stable_order,
    build_routing_v2,
    build_slot_v2,
)
from run_session_sandbox import (
    _classify_focus_reference,
    _display_trigger_focus_ref,
    _extract_named_person_label,
)


ROOT_DIR = Path(__file__).resolve().parent
V1_DIR = ROOT_DIR / "finetune_data"
V3_DIR = V1_DIR / "v3"


ROUTING_V3_EDGE_CASES: list[dict[str, Any]] = [
    {"label": "known_branch_new_plain", "node_id": "dark_known_branch", "user_reply": "das ist neu fuer mich", "intent": "new"},
    {"label": "known_branch_known_plain", "node_id": "dark_known_branch", "user_reply": "das kenne ich von frueher", "intent": "known"},
    {"label": "immediate_feeling_pressure", "node_id": "dark_scene_immediate_feeling", "user_reply": "druck", "intent": "ready"},
    {"label": "immediate_feeling_shame", "node_id": "dark_scene_immediate_feeling", "user_reply": "scham", "intent": "ready"},
    {"label": "happening_ready_plain", "node_id": "dark_scene_happening", "user_reply": "sie lachen mich aus", "intent": "ready"},
    {"label": "perception_audio_canonical", "node_id": "dark_scene_perception", "user_reply": "ich hoere sie lachen", "intent": "audio"},
    {"label": "perception_visual_canonical", "node_id": "dark_scene_perception", "user_reply": "ich sehe mehrere leute vor mir", "intent": "visual"},
    {"label": "perception_other_sense_canonical", "node_id": "dark_scene_perception", "user_reply": "ich rieche nur qualm", "intent": "other_sense"},
    {"label": "dark_who_question_plain", "node_id": "dark_scene_who", "user_reply": "was meinst du mit genau", "intent": "question"},
    {"label": "dark_who_ready_group", "node_id": "dark_scene_who", "user_reply": "eine gruppe jungs auf dem schulhof", "intent": "ready"},
    {"label": "mode_visual_people", "node_id": "dark_scene_mode_clarify", "user_reply": "ich sehe da leute", "intent": "visual"},
    {"label": "mode_other_sense_smoke", "node_id": "dark_scene_mode_clarify", "user_reply": "ich rieche qualm", "intent": "other_sense"},
    {"label": "hell_level_hell_plain", "node_id": "hell_light_level", "user_reply": "eindeutig hell", "intent": "hell_light"},
    {"label": "hell_level_dark_plain", "node_id": "hell_light_level", "user_reply": "eher dunkel", "intent": "darker_or_other"},
    {"label": "hell_wait_resolving", "node_id": "hell_hypnose_wait", "user_reply": "es loest sich schon weiter", "intent": "resolving"},
    {"label": "hell_wait_more_time", "node_id": "hell_hypnose_wait", "user_reply": "ich brauche noch einen moment", "intent": "need_more_time"},
    {"label": "hell_wait_resolved", "node_id": "hell_hypnose_wait", "user_reply": "es ist schon aufgeloest", "intent": "resolved"},
    {"label": "group_multiple_name_ready", "node_id": "group_multiple_people_name", "user_reply": "anna", "intent": "ready"},
    {"label": "group_multiple_required_name_ready", "node_id": "group_multiple_required_name", "user_reply": "paul", "intent": "ready"},
    {"label": "group_person_ready_no_plain", "node_id": "group_person_ready", "user_reply": "noch nicht ganz", "intent": "no"},
    {"label": "group_person_reason_ready_plain", "node_id": "group_person_trigger_reason", "user_reply": "gruppenzwang", "intent": "ready"},
    {"label": "group_source_whole_group_plain", "node_id": "group_source_kind", "user_reply": "klar von allen zusammen", "intent": "whole_group"},
    {"label": "group_source_one_person_plain", "node_id": "group_source_kind", "user_reply": "eigentlich nur von einer person", "intent": "one_person"},
    {"label": "group_source_multiple_plain", "node_id": "group_source_kind", "user_reply": "von mehreren einzeln", "intent": "multiple_people"},
    {"label": "group_image_ready_question_plain", "node_id": "group_image_ready", "user_reply": "wie meinst du das genau", "intent": "question"},
    {"label": "group_image_ready_support_plain", "node_id": "group_image_ready", "user_reply": "das ist mir gerade zu viel", "intent": "support_needed"},
    {"label": "group_core_ready_short", "node_id": "group_person_trigger_core", "user_reply": "ich glaube es steht fuer ausgrenzung", "intent": "ready"},
    {"label": "origin_other_kind_laughter", "node_id": "origin_other_target_kind", "user_reply": "das lachen von denen", "intent": "other"},
    {"label": "origin_owner_self_plain", "node_id": "origin_cause_owner", "user_reply": "eher aus mir selbst", "intent": "self", "runtime_slots": {"trigger_focus_ref": "die clique am pausenhof"}},
    {"label": "origin_owner_external_plain", "node_id": "origin_cause_owner", "user_reply": "eher von den anderen", "intent": "someone_else", "runtime_slots": {"trigger_focus_ref": "die clique am pausenhof"}},
    {"label": "scene_access_other_sense_taste", "node_id": "scene_access_followup", "user_reply": "ich schmecke rauch", "intent": "nonvisual_access"},
    {"label": "scene_access_visual_dark_plain", "node_id": "scene_access_followup", "user_reply": "es wirkt eher dunkel", "intent": "visual_dark"},
    {"label": "feeling_intensity_compound", "node_id": "dark_scene_feeling_intensity", "user_reply": "es drueckt stark in der brust", "intent": "feeling_and_intensity"},
    {"label": "feeling_intensity_scale_phrase", "node_id": "dark_scene_feeling_intensity", "user_reply": "sehr stark, fast neun von zehn", "intent": "intensity_only"},
]


SLOT_V3_EDGE_CASES: dict[str, list[dict[str, Any]]] = {
    "focus_kind": [
        {"user_reply": "die clique", "expected": {"focus_kind": "group"}},
        {"user_reply": "ich glaube es sind die freunde", "expected": {"focus_kind": "group"}},
        {"user_reply": "peter", "expected": {"focus_kind": "person"}},
        {"user_reply": "das auslachen", "expected": {"focus_kind": "other"}},
        {"user_reply": "der geruch nach rauch", "expected": {"focus_kind": "other"}},
    ],
    "named_person_label": [
        {"user_reply": "peter", "expected": {"named_person": "Peter"}},
        {"user_reply": "paul", "expected": {"named_person": "Paul"}},
        {"user_reply": "mein chef", "expected": {"named_person": "Mein Chef"}},
        {"user_reply": "die clique", "expected": {"named_person": None}},
    ],
    "display_trigger_focus_ref": [
        {"trigger_focus_ref": "die clique", "expected": {"display_trigger_focus_ref": "diese Gruppe"}},
        {"trigger_focus_ref": "ich glaube es sind die freunde", "expected": {"display_trigger_focus_ref": "diese Gruppe"}},
        {"trigger_focus_ref": "peter", "expected": {"display_trigger_focus_ref": "Peter"}},
        {"trigger_focus_ref": "der geruch nach rauch", "expected": {"display_trigger_focus_ref": "der geruch nach rauch"}},
    ],
}


def build_routing_v3() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows, v2_review = build_routing_v2()
    edge_rows = []
    for index, case in enumerate(ROUTING_V3_EDGE_CASES, start=1):
        row = _routing_edge_record(index, case)
        row["source"] = ["gold_v3_edge_cases", "label_hardening"]
        row["id"] = f"routing-v3-{index:04d}"
        edge_rows.append(row)
    combined = base_rows + edge_rows
    deduped, duplicates_removed = _dedupe_records(combined)
    review = {
        "v2_count": len(base_rows),
        "edge_case_count_added": len(edge_rows),
        "duplicates_removed": duplicates_removed,
        "v3_count": len(deduped),
        "edge_labels": [case["label"] for case in ROUTING_V3_EDGE_CASES],
        "top_node_counts_v3": Counter(row["input"]["node_id"] for row in deduped).most_common(15),
        "v2_reference": v2_review,
    }
    return deduped, review


def build_slot_v3() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows, v2_review = build_slot_v2()
    edge_rows: list[dict[str, Any]] = []
    helper_alignment: list[dict[str, Any]] = []
    idx = 0

    for task, cases in SLOT_V3_EDGE_CASES.items():
        for case in cases:
            idx += 1
            row = _slot_edge_record(idx, task, case)
            row["source"] = ["gold_v3_edge_cases", "label_hardening"]
            row["id"] = f"slot-v3-{idx:04d}"
            edge_rows.append(row)
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
        "v2_count": len(base_rows),
        "edge_case_count_added": len(edge_rows),
        "duplicates_removed": duplicates_removed,
        "v3_count": len(deduped),
        "task_counts_v3": Counter(row["task"] for row in deduped),
        "helper_alignment_total": len(helper_alignment),
        "helper_alignment_mismatches": len(mismatches),
        "helper_alignment_examples": mismatches[:20],
        "v2_reference": v2_review,
    }
    return deduped, review


def build_v3_splits(rows: list[dict[str, Any]], *, dataset: str) -> dict[str, list[dict[str, Any]]]:
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


def write_v3_artifacts() -> dict[str, Any]:
    V3_DIR.mkdir(parents=True, exist_ok=True)

    routing_rows, routing_review = build_routing_v3()
    slot_rows, slot_review = build_slot_v3()

    _write_jsonl(V3_DIR / "routing_gold_v3.jsonl", routing_rows)
    _write_jsonl(V3_DIR / "slot_extraction_gold_v3.jsonl", slot_rows)

    routing_split = build_v3_splits(routing_rows, dataset="routing")
    slot_split = build_v3_splits(slot_rows, dataset="slot_extraction")
    _write_jsonl(V3_DIR / "routing_train_v3.jsonl", routing_split["train"])
    _write_jsonl(V3_DIR / "routing_eval_v3.jsonl", routing_split["eval"])
    _write_jsonl(V3_DIR / "slot_extraction_train_v3.jsonl", slot_split["train"])
    _write_jsonl(V3_DIR / "slot_extraction_eval_v3.jsonl", slot_split["eval"])

    review = {
        "version": 3,
        "datasets": {
            "routing": routing_review,
            "slot_extraction": slot_review,
        },
        "split_counts": {
            "routing_train_v3.jsonl": len(routing_split["train"]),
            "routing_eval_v3.jsonl": len(routing_split["eval"]),
            "slot_extraction_train_v3.jsonl": len(slot_split["train"]),
            "slot_extraction_eval_v3.jsonl": len(slot_split["eval"]),
        },
    }
    (V3_DIR / "gold_v3_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (V3_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Gold V3",
                "",
                "Gold V3 haertet `routing` und `slot_extraction` weiter in Richtung kanonischer Label-Ontologie.",
                "",
                "Ziele:",
                "- Synonym-Drift wie `first_time`, `see`, `hear`, `smell`, `unknown` weiter zurueckdruecken",
                "- mehr kurze, knappe, aber inhaltlich gueltige Antworten auf den richtigen kanonischen Intent abbilden",
                "- Gruppen-/Person-/Other-Slots weiter gegen umgangssprachliche Formulierungen absichern",
                "",
                "Dateien:",
                "- `routing_gold_v3.jsonl`",
                "- `routing_train_v3.jsonl`",
                "- `routing_eval_v3.jsonl`",
                "- `slot_extraction_gold_v3.jsonl`",
                "- `slot_extraction_train_v3.jsonl`",
                "- `slot_extraction_eval_v3.jsonl`",
                "- `gold_v3_review.json`",
                "",
                "Generator:",
                "- `backend/build_gold_v3_datasets.py`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return review


def main() -> None:
    review = write_v3_artifacts()
    print(json.dumps(review["split_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
