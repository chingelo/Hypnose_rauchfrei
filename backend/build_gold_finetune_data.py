from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_session_sandbox import (
    INACTIVITY_END_TEXT,
    INACTIVITY_WARNING_TEXT,
    QUESTION_ANSWER_HINTS,
    _classify_focus_reference,
    _display_trigger_focus_ref,
    _empty_input_reply,
    _extract_named_person_label,
    _question_announcement_reply,
    _render_runtime_question,
    _render_runtime_text,
)
from session_sandbox_orchestrator import (
    ScriptNodeSpec,
    available_node_ids,
    get_node_spec,
)


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "finetune_data"

META_INTENTS = {"unclear", "repeat", "abort", "question", "support_needed"}

DEFAULT_RUNTIME_SLOTS: dict[str, str] = {
    "named_person": "Peter",
    "trigger_focus_ref": "die menschen wo rauchen",
    "group_person_trigger_reason": "gruppendruck",
    "group_person_trigger_role": "von allen",
}

GENERIC_INTENT_EXAMPLES: dict[str, list[str]] = {
    "abort": ["ich will abbrechen"],
    "audio": ["ich hoere lachen"],
    "both": ["beides"],
    "continue": ["ja wir koennen weiter"],
    "dark_or_both_or_quieter": ["deutlich ruhiger"],
    "darker_or_other": ["dunkel"],
    "distance": ["mehr Abstand"],
    "feeling_and_intensity": ["ich spuere Druck, sehr stark"],
    "focus": ["klarerer Fokus"],
    "group": ["eine Gruppe"],
    "hell_light": ["hell"],
    "intensity_only": ["sehr stark"],
    "known": ["ich kenne das aus frueheren Momenten"],
    "less_brightness": ["weniger Helligkeit"],
    "multiple_people": ["von mehreren einzelnen Personen"],
    "multiple_required": ["wir muessen mehrere Personen einzeln einbeziehen"],
    "need_more_time": ["ich brauche noch einen Moment"],
    "new": ["zum ersten Mal"],
    "no": ["noch nicht"],
    "nonvisual_access": ["ich rieche Rauch"],
    "nothing": ["gar nichts"],
    "nothing_yet": ["noch nichts"],
    "one_person": ["von Peter"],
    "other": ["etwas anderes in der Situation"],
    "other_sense": ["ich spuere Druck in der Brust"],
    "person": ["eine bestimmte Person"],
    "pleasant": ["sehr angenehm"],
    "provided_scale": ["8"],
    "question": ["wie meinst du das?"],
    "repeat": ["wiederhole die Frage bitte"],
    "representative_enough": ["eine stellvertretende Person reicht"],
    "resolved": ["es hat sich bereits aufgeloest"],
    "resolving": ["es loest sich noch auf"],
    "self": ["eher in mir selbst"],
    "someone_else": ["eher bei jemand anderem"],
    "still_hell": ["immer noch hell"],
    "support_needed": ["es ist mir gerade zu viel"],
    "technical_issue": ["ich hoere dich nicht gut"],
    "unclear": ["banane"],
    "unpleasant": ["unangenehm"],
    "visual": ["ich sehe eine Gruppe auf dem Schulhof"],
    "visual_dark": ["ich sehe etwas Dunkles"],
    "visual_hell": ["ich sehe etwas Helles"],
    "whole_group": ["von der ganzen Gruppe"],
    "yes": ["ja"],
}

READY_EXAMPLES_BY_NODE: dict[str, list[str]] = {
    "dark_scene_age": ["18"],
    "dark_scene_audio_detail": ["sie lachen mich aus"],
    "dark_scene_first_spuerbar": ["Enge in der Brust"],
    "dark_scene_happening": ["sie lachen und halten mir eine Zigarette hin"],
    "dark_scene_immediate_feeling": ["Druck"],
    "dark_scene_other_sense": ["kalter Rauch und Enge in der Brust"],
    "dark_scene_people_who": ["Peter und zwei andere aus der Klasse"],
    "dark_scene_who": ["eine Gruppe Kinder auf dem Schulhof"],
    "group_multiple_people_name": ["Anna"],
    "group_multiple_required_name": ["Peter"],
    "group_next_person_name": ["Hansi"],
    "group_person_trigger_core": ["sie lachen, weil ich der Einzige bin, der nicht raucht"],
    "group_person_trigger_reason": ["Gruppendruck"],
    "group_person_trigger_role": ["er steht fuer die Gruppe"],
    "group_representative_name": ["Peter"],
    "group_specific_person_name": ["Peter"],
    "origin_trigger_source": ["die menschen wo rauchen"],
    "person_switch_why": ["ich wollte einfach dazugehören"],
    "phase4_common_done_signal": ["ich bin wieder im Sessel"],
    "phase4_common_feel_after_aversion": ["deutlich ruhiger"],
    "phase4_common_feel_after_learning": ["leichter und klarer"],
}

FOCUS_KIND_CASES: dict[str, list[str]] = {
    "person": [
        "peter",
        "paul",
        "mein vater",
        "meine mutter",
        "der lehrer",
        "die lehrerin",
        "mein chef",
        "meine chefin",
        "mein freund",
        "meine freundin",
        "der arzt",
        "die therapeutin",
    ],
    "group": [
        "die menschen wo rauchen",
        "die clique auf dem pausenhof",
        "meine freunde",
        "die gruppe vor der schule",
        "die klasse",
        "alle anderen",
        "mehrere jungs aus der klasse",
        "die leute in der raucherecke",
        "die kinder auf dem schulhof",
        "meine familie",
        "die eltern",
        "ein paar freundinnen",
    ],
    "other": [
        "der rauch",
        "der qualm",
        "der geruch",
        "die zigarette",
        "die situation auf dem schulhof",
        "dieser moment",
        "das lachen",
        "das verhalten",
        "der blick",
        "die pause auf dem schulhof",
        "die raucherecke",
        "das ereignis in der kueche",
        "die farbe rot",
        "der geruch von rauch",
    ],
}

NAMED_PERSON_CASES: dict[str, str | None] = {
    "peter": "Peter",
    "paul": "Paul",
    "das ist paul": "Paul",
    "mein vater": "Mein Vater",
    "die lehrerin": "Lehrerin",
    "der chef": "Chef",
    "die menschen wo rauchen": None,
    "die clique auf dem pausenhof": None,
    "meine freunde": None,
    "der rauch": None,
    "der geruch": None,
    "die situation auf dem schulhof": None,
    "das lachen": None,
    "die raucherecke": None,
}

DISPLAY_FOCUS_CASES: dict[str, str] = {
    "die menschen wo rauchen": "diese Gruppe",
    "die clique auf dem pausenhof": "diese Gruppe",
    "ich glaube die freunde die dort am rauchen sind und mich auslachen": "diese Gruppe",
    "ich glaube es sind die freunde": "diese Gruppe",
    "die situation auf dem schulhof": "die situation auf dem schulhof",
    "der rauch": "der rauch",
    "Peter": "Peter",
}


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _render_question(node_id: str, runtime_slots: dict[str, str]) -> str:
    return _render_runtime_question(node_id, runtime_slots)


def _render_reply(template: str, runtime_slots: dict[str, str]) -> str:
    return _render_runtime_text(template, runtime_slots)


def _semantic_specs() -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    for node_id in sorted(available_node_ids()):
        spec = get_node_spec(node_id)
        if not isinstance(spec, ScriptNodeSpec):
            specs.append((node_id, spec))
    return specs


def _content_intents(spec: Any) -> list[str]:
    return [intent for intent in spec.allowed_intents if intent not in META_INTENTS]


def _runtime_slots_for(node_id: str) -> dict[str, str]:
    slots = dict(DEFAULT_RUNTIME_SLOTS)
    if node_id in {
        "origin_cause_owner",
        "origin_other_target_kind",
        "origin_trigger_source",
        "group_image_ready",
        "group_source_kind",
        "group_whole_scope",
    }:
        slots["trigger_focus_ref"] = "die menschen wo rauchen"
    return slots


def _examples_for_intent(node_id: str, intent: str) -> list[str]:
    if intent == "ready":
        examples = READY_EXAMPLES_BY_NODE.get(node_id)
        if not examples:
            raise KeyError(f"Missing ready example for node {node_id}")
        return examples
    examples = GENERIC_INTENT_EXAMPLES.get(intent)
    if not examples:
        raise KeyError(f"Missing generic example for intent {intent} on node {node_id}")
    return examples


def build_routing_gold() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    idx = 0
    for node_id, spec in _semantic_specs():
        runtime_slots = _runtime_slots_for(node_id)
        question_text = _render_question(node_id, runtime_slots)
        session_hint = QUESTION_ANSWER_HINTS.get(node_id, "")

        for intent in spec.allowed_intents:
            examples = _examples_for_intent(node_id, intent)
            rule = spec.routing_rules[intent]
            for example in examples:
                idx += 1
                records.append(
                    {
                        "id": f"routing-{idx:04d}",
                        "task": "routing",
                        "source": ["current_runtime", "semantic_spec"],
                        "input": {
                            "node_id": node_id,
                            "question_text": question_text,
                            "runtime_slots": runtime_slots,
                            "session_hint": session_hint,
                            "user_reply": example,
                        },
                        "output": {
                            "intent": intent,
                            "action": rule["action"],
                            "next_node": rule["next_node"],
                        },
                    }
                )
    return records


def build_slot_extraction_gold() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    idx = 0

    for expected_kind, phrases in FOCUS_KIND_CASES.items():
        for phrase in phrases:
            idx += 1
            records.append(
                {
                    "id": f"slot-{idx:04d}",
                    "task": "focus_kind",
                    "source": ["run_session_sandbox", "test_matrix"],
                    "input": {"user_reply": phrase},
                    "output": {"focus_kind": _classify_focus_reference(phrase)},
                }
            )

    for phrase, expected in NAMED_PERSON_CASES.items():
        idx += 1
        records.append(
            {
                "id": f"slot-{idx:04d}",
                "task": "named_person_label",
                "source": ["run_session_sandbox", "test_matrix"],
                "input": {"user_reply": phrase},
                "output": {"named_person": _extract_named_person_label(phrase)},
            }
        )

    for phrase, expected in DISPLAY_FOCUS_CASES.items():
        idx += 1
        records.append(
            {
                "id": f"slot-{idx:04d}",
                "task": "display_trigger_focus_ref",
                "source": ["run_session_sandbox", "runtime_rendering"],
                "input": {"trigger_focus_ref": phrase},
                "output": {"display_trigger_focus_ref": _display_trigger_focus_ref(phrase)},
            }
        )

    return records


def build_clarification_gold() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    idx = 0
    for node_id, spec in _semantic_specs():
        runtime_slots = _runtime_slots_for(node_id)
        question_text = _render_question(node_id, runtime_slots)

        for intent in ("question", "unclear", "repeat"):
            reply_template = spec.same_node_replies.get(intent)
            if not reply_template:
                continue
            idx += 1
            records.append(
                {
                    "id": f"clarification-{idx:04d}",
                    "task": "same_node_reply",
                    "source": ["semantic_spec", "runtime_rendering"],
                    "input": {
                        "node_id": node_id,
                        "question_text": question_text,
                        "runtime_slots": runtime_slots,
                        "trigger": intent,
                    },
                    "output": {"reply_text": _render_reply(reply_template, runtime_slots)},
                }
            )

        for attempt in (0, 1):
            idx += 1
            records.append(
                {
                    "id": f"clarification-{idx:04d}",
                    "task": "empty_input_reply",
                    "source": ["run_session_sandbox", "silence_policy"],
                    "input": {
                        "node_id": node_id,
                        "question_text": question_text,
                        "runtime_slots": runtime_slots,
                        "clarify_attempt": attempt,
                        "trigger": "silence",
                    },
                    "output": {"reply_text": _empty_input_reply(node_id, attempt, runtime_slots)},
                }
            )

    return records


def build_support_abort_gold() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    idx = 0
    for node_id, spec in _semantic_specs():
        runtime_slots = _runtime_slots_for(node_id)
        question_text = _render_question(node_id, runtime_slots)

        for intent in ("support_needed", "abort"):
            reply_template = spec.same_node_replies.get(intent)
            if not reply_template:
                continue
            idx += 1
            records.append(
                {
                    "id": f"support-{idx:04d}",
                    "task": intent,
                    "source": ["semantic_spec", "runtime_rendering"],
                    "input": {
                        "node_id": node_id,
                        "question_text": question_text,
                        "runtime_slots": runtime_slots,
                        "trigger": intent,
                    },
                    "output": {"reply_text": _render_reply(reply_template, runtime_slots)},
                }
            )

    for node_id in ("session_phase2_ready", "hell_light_level"):
        question_text = _render_question(node_id, DEFAULT_RUNTIME_SLOTS)
        idx += 1
        records.append(
            {
                "id": f"support-{idx:04d}",
                "task": "inactivity_warning",
                "source": ["run_session_sandbox", "silence_policy"],
                "input": {
                    "node_id": node_id,
                    "question_text": question_text,
                    "clarify_attempt": 4,
                    "trigger": "silence",
                },
                "output": {"reply_text": INACTIVITY_WARNING_TEXT},
            }
        )
        idx += 1
        records.append(
            {
                "id": f"support-{idx:04d}",
                "task": "inactivity_end",
                "source": ["run_session_sandbox", "silence_policy"],
                "input": {
                    "node_id": node_id,
                    "question_text": question_text,
                    "clarify_attempt": 5,
                    "trigger": "silence",
                },
                "output": {"reply_text": INACTIVITY_END_TEXT},
            }
        )

    for attempt in range(3):
        idx += 1
        records.append(
            {
                "id": f"support-{idx:04d}",
                "task": "question_announcement_reply",
                "source": ["run_session_sandbox"],
                "input": {
                    "trigger": "question_announcement",
                    "clarify_attempt": attempt,
                },
                "output": {"reply_text": _question_announcement_reply(attempt)},
            }
        )

    return records


def build_all_datasets() -> dict[str, list[dict[str, Any]]]:
    return {
        "routing_gold.jsonl": build_routing_gold(),
        "slot_extraction_gold.jsonl": build_slot_extraction_gold(),
        "clarification_gold.jsonl": build_clarification_gold(),
        "support_abort_gold.jsonl": build_support_abort_gold(),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    _ensure_output_dir()
    datasets = build_all_datasets()
    for name, records in datasets.items():
        write_jsonl(OUTPUT_DIR / name, records)
        print(f"{name}: {len(records)}")


if __name__ == "__main__":
    main()
