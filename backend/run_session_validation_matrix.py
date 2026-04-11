from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from live_api_guard import DEFAULT_APPROVAL_FILE, LiveApiCallBudget, build_live_api_budget
from run_session_sandbox import (
    SCENARIOS,
    _handle_silence,
    _render_runtime_question,
    parse_json_object,
    resolve_openai_client,
    run_batch,
)
from session_sandbox_orchestrator import (
    ScriptNodeSpec,
    SemanticNodeSpec,
    available_node_ids,
    build_request,
    get_node_spec,
    repair_semantic_payload,
    validate_semantic_decision,
)


@dataclass(frozen=True)
class BranchCase:
    label: str
    node_id: str
    customer_message: str
    expected_intent: str
    expected_next_node: str
    runtime_slots: dict[str, str] = field(default_factory=dict)
    session_context: str = ""


@dataclass
class ValidationFailure:
    suite: str
    label: str
    node_id: str
    customer_message: str
    expected: dict[str, Any]
    actual: dict[str, Any]


def _slots(**kwargs: str) -> dict[str, str]:
    return dict(kwargs)


COMMON_GROUP_SLOTS = _slots(trigger_focus_ref="die Gruppe", named_person="Peter")
COMMON_PERSON_SLOTS = _slots(named_person="Peter")
GROUP_REASON_SLOTS = _slots(named_person="Peter", group_person_trigger_reason="er lacht mich aus")
GROUP_CORE_SLOTS = _slots(
    named_person="Peter",
    group_person_trigger_reason="er lacht mich aus",
    group_person_trigger_role="von allen",
)
PERSON_SWITCH_SLOTS = _slots(named_person="Peter")
ORIGIN_GROUP_SLOTS = _slots(trigger_focus_ref="die Gruppe")
ORIGIN_PERSON_SLOTS = _slots(trigger_focus_ref="Peter", named_person="Peter")


BRANCH_CASES: list[BranchCase] = [
    BranchCase("phase1_preflight_continue", "session_phase1_preflight_check", "ja ich bin bereit", "continue", "session_phase1_setup_script"),
    BranchCase("phase1_preflight_technical", "session_phase1_preflight_check", "mein ton geht nicht", "technical_issue", "session_phase1_preflight_check"),
    BranchCase("phase1_setup_continue", "session_phase1_anchor_after_setup", "ja passt jetzt", "continue", "session_phase1_mindset_script"),
    BranchCase("phase1_setup_technical", "session_phase1_anchor_after_setup", "ich habe probleme mit dem ton", "technical_issue", "session_phase1_anchor_after_setup"),
    BranchCase("phase1_focus_continue", "session_phase1_anchor_before_focus", "ja wir koennen weiter", "continue", "session_phase1_focus_script"),
    BranchCase("phase1_focus_technical", "session_phase1_anchor_before_focus", "die verbindung ruckelt", "technical_issue", "session_phase1_anchor_before_focus"),
    BranchCase("phase2_ready_yes", "session_phase2_ready", "ja", "yes", "session_phase2_post_ready_script"),
    BranchCase("phase2_ready_no", "session_phase2_ready", "noch nicht", "no", "session_phase2_ready"),
    BranchCase("phase2_eyes_yes", "session_phase2_eyes_closed", "ja", "yes", "session_phase2_post_eyes_script"),
    BranchCase("phase2_eyes_no", "session_phase2_eyes_closed", "noch nicht", "no", "session_phase2_eyes_closed"),
    BranchCase("phase2_scene_yes", "session_phase2_scene_found", "ja", "yes", "session_phase2_post_scene_script"),
    BranchCase("phase2_scene_no", "session_phase2_scene_found", "noch nicht", "no", "session_phase2_scene_found"),
    BranchCase("phase2_feel_yes", "session_phase2_feel_clear", "ja deutlich", "yes", "session_phase2_post_feel_script"),
    BranchCase("phase2_feel_no", "session_phase2_feel_clear", "noch nicht klar", "no", "session_phase2_feel_clear"),
    BranchCase("phase2_scale_clear_yes", "session_phase2_scale_clear", "ja", "yes", "session_phase2_scale_before"),
    BranchCase("phase2_scale_clear_no", "session_phase2_scale_clear", "noch nicht", "no", "session_phase2_scale_clear"),
    BranchCase("phase2_scale_before", "session_phase2_scale_before", "8", "provided_scale", "session_phase2_post_scale_before_script"),
    BranchCase("phase2_scale_after", "session_phase2_scale_after", "5", "provided_scale", "session_phase2_post_scale_after_script"),
    BranchCase("phase2_continue_yes", "session_phase2_continue_to_main", "ja", "yes", "session_phase2_end_script"),
    BranchCase("phase2_continue_no", "session_phase2_continue_to_main", "noch nicht", "no", "session_phase2_continue_to_main"),
    BranchCase("phase4_hell_light", "hell_light_level", "hell", "hell_light", "hell_feel_branch"),
    BranchCase("phase4_dark_light", "hell_light_level", "dunkel", "darker_or_other", "dark_scene_perception"),
    BranchCase("phase4_both_light", "hell_light_level", "beides", "both", "dark_follow_darker_intro"),
    BranchCase("scene_access_visual_hell", "scene_access_followup", "ich sehe etwas Helles", "visual_hell", "dark_scene_who"),
    BranchCase("scene_access_visual_dark", "scene_access_followup", "ich sehe eine dunkle Szene", "visual_dark", "dark_scene_who"),
    BranchCase("scene_access_nonvisual", "scene_access_followup", "ich spuere etwas im koerper", "nonvisual_access", "scene_access_body_bridge_intro"),
    BranchCase("scene_access_nothing", "scene_access_followup", "noch nichts", "nothing_yet", "scene_access_body_bridge_intro"),
    BranchCase("hell_feel_pleasant", "hell_feel_branch", "sehr angenehm", "pleasant", "hell_hypnose_loch_intro"),
    BranchCase("hell_feel_unpleasant", "hell_feel_branch", "unangenehm", "unpleasant", "hell_regulation_choice"),
    BranchCase("hell_wait_resolved", "hell_hypnose_wait", "aufgeloest", "resolved", "hell_post_resolved_terminal"),
    BranchCase("hell_wait_resolving", "hell_hypnose_wait", "loest sich noch", "resolving", "hell_hypnose_wait"),
    BranchCase("hell_wait_need_more_time", "hell_hypnose_wait", "ich brauche noch einen moment", "need_more_time", "hell_hypnose_wait"),
    BranchCase("hell_reg_distance", "hell_regulation_choice", "mehr abstand", "distance", "hell_regulation_check"),
    BranchCase("hell_reg_less_brightness", "hell_regulation_choice", "weniger helligkeit", "less_brightness", "hell_regulation_check"),
    BranchCase("hell_reg_focus", "hell_regulation_choice", "klarerer fokus", "focus", "hell_regulation_check"),
    BranchCase("hell_reg_check_still_hell", "hell_regulation_check", "immer noch hell", "still_hell", "hell_feel_branch"),
    BranchCase("hell_reg_check_darker", "hell_regulation_check", "jetzt eher dunkel", "dark_or_both_or_quieter", "dark_known_branch"),
    BranchCase("dark_scene_visual", "dark_scene_perception", "ich sehe eine szene", "visual", "dark_scene_who"),
    BranchCase("dark_scene_audio", "dark_scene_perception", "lachen", "audio", "dark_scene_audio_detail"),
    BranchCase("dark_scene_both", "dark_scene_perception", "ich sehe und hoere etwas", "both", "dark_scene_who"),
    BranchCase("dark_scene_other_sense", "dark_scene_perception", "druck", "other_sense", "dark_scene_other_sense"),
    BranchCase("dark_scene_nothing", "dark_scene_perception", "noch nichts", "nothing", "dark_scene_mode_clarify"),
    BranchCase("dark_mode_visual", "dark_scene_mode_clarify", "ich sehe jemanden", "visual", "dark_scene_who"),
    BranchCase("dark_mode_audio", "dark_scene_mode_clarify", "lachen", "audio", "dark_scene_audio_detail"),
    BranchCase("dark_mode_both", "dark_scene_mode_clarify", "ich sehe und hoere", "both", "dark_scene_who"),
    BranchCase("dark_mode_other", "dark_scene_mode_clarify", "druck", "other_sense", "dark_scene_other_sense"),
    BranchCase("dark_mode_nothing", "dark_scene_mode_clarify", "noch nichts", "nothing", "dark_scene_other_sense"),
    BranchCase("dark_who", "dark_scene_who", "pausenhof", "ready", "dark_scene_happening"),
    BranchCase("dark_audio_detail", "dark_scene_audio_detail", "lachen", "ready", "dark_scene_happening"),
    BranchCase("dark_other_sense", "dark_scene_other_sense", "druck", "ready", "dark_scene_first_spuerbar"),
    BranchCase("dark_first_spuerbar", "dark_scene_first_spuerbar", "enge", "ready", "dark_scene_happening"),
    BranchCase("dark_people_who", "dark_scene_people_who", "peter", "ready", "dark_scene_age"),
    BranchCase("dark_happening", "dark_scene_happening", "auslachen", "ready", "origin_trigger_source"),
    BranchCase("dark_age", "dark_scene_age", "12", "ready", "dark_scene_feeling_intensity"),
    BranchCase("dark_feeling_intensity_only", "dark_scene_feeling_intensity", "sehr stark", "intensity_only", "dark_scene_immediate_feeling"),
    BranchCase("dark_feeling_intensity_with_feeling", "dark_scene_feeling_intensity", "ich spuere druck in der brust", "feeling_and_intensity", "dark_known_branch"),
    BranchCase("dark_immediate_feeling", "dark_scene_immediate_feeling", "druck", "ready", "dark_known_branch"),
    BranchCase("dark_known", "dark_known_branch", "bekannt", "known", "dark_backtrace_terminal"),
    BranchCase("dark_new", "dark_known_branch", "zum ersten mal", "new", "dark_origin_terminal"),
    BranchCase("origin_trigger_ready", "origin_trigger_source", "die gruppe", "ready", "origin_trigger_known_branch", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_trigger_person", "origin_trigger_source", "mein vater", "ready", "origin_trigger_known_branch", _slots(trigger_focus_ref="mein vater")),
    BranchCase("origin_trigger_other_smell", "origin_trigger_source", "der geruch von rauch", "ready", "origin_trigger_known_branch", _slots(trigger_focus_ref="der geruch von rauch")),
    BranchCase("origin_trigger_other_color", "origin_trigger_source", "die farbe rot", "ready", "origin_trigger_known_branch", _slots(trigger_focus_ref="die farbe rot")),
    BranchCase("origin_trigger_other_event", "origin_trigger_source", "das auslachen", "ready", "origin_trigger_known_branch", _slots(trigger_focus_ref="das auslachen")),
    BranchCase("origin_trigger_other_situation", "origin_trigger_source", "die situation auf dem schulhof", "ready", "origin_trigger_known_branch", _slots(trigger_focus_ref="die situation auf dem schulhof")),
    BranchCase("origin_trigger_known_known", "origin_trigger_known_branch", "das kenne ich schon", "known", "dark_backtrace_terminal", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_trigger_known_new", "origin_trigger_known_branch", "zum ersten mal", "new", "origin_cause_owner", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_scene_relevance_here", "origin_scene_relevance", "das muessen wir hier loesen", "resolve_here", "origin_cause_owner", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_scene_relevance_backtrace", "origin_scene_relevance", "das fuehrt mich noch weiter zurueck", "older_origin", "dark_backtrace_terminal", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_cause_self", "origin_cause_owner", "eher in mir", "self", "origin_self_resolution_intro", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_cause_other", "origin_cause_owner", "jemand anderes", "someone_else", "origin_other_target_kind", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_kind_group", "origin_other_target_kind", "gruppe", "group", "group_branch_intro", ORIGIN_GROUP_SLOTS),
    BranchCase("origin_kind_person", "origin_other_target_kind", "peter", "person", "origin_person_branch_intro", ORIGIN_PERSON_SLOTS),
    BranchCase("origin_kind_person_role", "origin_other_target_kind", "mein vater", "person", "origin_person_branch_intro", _slots(trigger_focus_ref="mein vater", named_person="Mein Vater")),
    BranchCase("origin_kind_group_friends", "origin_other_target_kind", "die freunde auf dem pausenhof", "group", "group_branch_intro", _slots(trigger_focus_ref="die freunde auf dem pausenhof")),
    BranchCase("origin_kind_group_smokers", "origin_other_target_kind", "die menschen wo rauchen", "group", "group_branch_intro", _slots(trigger_focus_ref="die menschen wo rauchen")),
    BranchCase("origin_kind_other_smell", "origin_other_target_kind", "der geruch von rauch", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="der geruch von rauch")),
    BranchCase("origin_kind_other_color", "origin_other_target_kind", "die farbe rot", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="die farbe rot")),
    BranchCase("origin_kind_other_behavior", "origin_other_target_kind", "das verhalten", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="das verhalten")),
    BranchCase("origin_kind_other_event", "origin_other_target_kind", "das auslachen", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="das auslachen")),
    BranchCase("origin_kind_other_gaze", "origin_other_target_kind", "der blick", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="der blick")),
    BranchCase("origin_kind_other_situation", "origin_other_target_kind", "die situation auf dem schulhof", "other", "origin_self_resolution_intro", _slots(trigger_focus_ref="die situation auf dem schulhof")),
    BranchCase("origin_self_need_ready", "origin_self_need", "schutz und halt", "ready", "origin_self_release_intro", _slots(trigger_focus_ref="der druck")),
    BranchCase("group_image_yes", "group_image_ready", "ja", "yes", "group_source_kind", COMMON_GROUP_SLOTS),
    BranchCase("group_image_no", "group_image_ready", "noch nicht", "no", "group_image_ready", COMMON_GROUP_SLOTS),
    BranchCase("group_source_whole", "group_source_kind", "die ganze gruppe", "whole_group", "group_whole_scope", COMMON_GROUP_SLOTS),
    BranchCase("group_source_person", "group_source_kind", "peter", "one_person", "group_specific_person_intro", COMMON_GROUP_SLOTS),
    BranchCase("group_source_multiple", "group_source_kind", "peter und anna", "multiple_people", "group_multiple_people_intro", COMMON_GROUP_SLOTS),
    BranchCase("group_scope_representative", "group_whole_scope", "eine reicht", "representative_enough", "group_select_representative_intro", COMMON_GROUP_SLOTS),
    BranchCase("group_scope_multiple_required", "group_whole_scope", "mehrere sind noetig", "multiple_required", "group_multiple_required_intro", COMMON_GROUP_SLOTS),
    BranchCase("group_rep_name", "group_representative_name", "peter", "ready", "group_bring_person_forward", COMMON_GROUP_SLOTS),
    BranchCase("group_specific_name", "group_specific_person_name", "peter", "ready", "group_bring_person_forward", COMMON_GROUP_SLOTS),
    BranchCase("group_multiple_name", "group_multiple_people_name", "anna", "ready", "group_bring_person_forward", COMMON_GROUP_SLOTS),
    BranchCase("group_multiple_required_name", "group_multiple_required_name", "anna", "ready", "group_bring_person_forward", COMMON_GROUP_SLOTS),
    BranchCase("group_person_ready_yes", "group_person_ready", "ja", "yes", "group_person_handoff", COMMON_PERSON_SLOTS),
    BranchCase("group_person_ready_no", "group_person_ready", "noch nicht", "no", "group_person_ready", COMMON_PERSON_SLOTS),
    BranchCase("group_trigger_reason", "group_person_trigger_reason", "auslachen", "ready", "group_person_trigger_role", COMMON_PERSON_SLOTS),
    BranchCase("group_trigger_role", "group_person_trigger_role", "von allen", "ready", "group_person_trigger_core", GROUP_REASON_SLOTS),
    BranchCase("group_trigger_core", "group_person_trigger_core", "dazugehoeren", "ready", "person_switch_ready_intro", GROUP_CORE_SLOTS),
    BranchCase("person_switch_ready_yes", "person_switch_ready", "ja", "yes", "person_switch_intro", PERSON_SWITCH_SLOTS),
    BranchCase("person_switch_ready_no", "person_switch_ready", "noch nicht", "no", "person_switch_ready", PERSON_SWITCH_SLOTS),
    BranchCase("group_next_person_yes", "group_next_person_check", "ja", "yes", "group_next_person_name", PERSON_SWITCH_SLOTS),
    BranchCase("group_next_person_no", "group_next_person_check", "nein", "no", "group_resolution_complete", PERSON_SWITCH_SLOTS),
    BranchCase("group_next_person_name", "group_next_person_name", "anna", "ready", "group_bring_person_forward", COMMON_GROUP_SLOTS),
    BranchCase("person_hears_yes", "person_switch_hears", "ja", "yes", "person_switch_sees_customer", PERSON_SWITCH_SLOTS),
    BranchCase("person_hears_no", "person_switch_hears", "noch nicht", "no", "person_switch_hears", PERSON_SWITCH_SLOTS),
    BranchCase("person_sees_customer_yes", "person_switch_sees_customer", "ja", "yes", "person_switch_sees_impact", PERSON_SWITCH_SLOTS),
    BranchCase("person_sees_customer_no", "person_switch_sees_customer", "noch nicht", "no", "person_switch_sees_customer", PERSON_SWITCH_SLOTS),
    BranchCase("person_sees_impact_yes", "person_switch_sees_impact", "ja", "yes", "person_switch_heard_customer", PERSON_SWITCH_SLOTS),
    BranchCase("person_sees_impact_no", "person_switch_sees_impact", "noch nicht", "no", "person_switch_sees_impact", PERSON_SWITCH_SLOTS),
    BranchCase("person_heard_customer_yes", "person_switch_heard_customer", "ja", "yes", "person_switch_why", PERSON_SWITCH_SLOTS),
    BranchCase("person_heard_customer_no", "person_switch_heard_customer", "noch nicht", "no", "person_switch_heard_customer", PERSON_SWITCH_SLOTS),
    BranchCase("person_switch_why_ready", "person_switch_why", "ueberforderung", "ready", "person_switch_aware_trigger", PERSON_SWITCH_SLOTS),
    BranchCase("person_aware_trigger_yes", "person_switch_aware_trigger", "ja", "yes", "person_switch_return_intro", PERSON_SWITCH_SLOTS),
    BranchCase("person_aware_trigger_no", "person_switch_aware_trigger", "noch nicht", "no", "person_switch_aware_trigger", PERSON_SWITCH_SLOTS),
    BranchCase("person_self_heard_yes", "person_switch_self_heard", "ja", "yes", "person_switch_self_understands", PERSON_SWITCH_SLOTS),
    BranchCase("person_self_heard_no", "person_switch_self_heard", "noch nicht", "no", "person_switch_self_heard", PERSON_SWITCH_SLOTS),
    BranchCase("person_self_understands_yes", "person_switch_self_understands", "ja", "yes", "group_resolution_complete", PERSON_SWITCH_SLOTS),
    BranchCase("person_self_understands_no", "person_switch_self_understands", "noch nicht", "no", "person_switch_self_understands", PERSON_SWITCH_SLOTS),
    BranchCase("common_sees_younger_yes", "phase4_common_sees_younger_self", "ja", "yes", "phase4_common_explain_to_younger"),
    BranchCase("common_sees_younger_no", "phase4_common_sees_younger_self", "noch nicht", "no", "phase4_common_sees_younger_self"),
    BranchCase("common_understood_yes", "phase4_common_understood", "ja", "yes", "phase4_common_first_cigarette"),
    BranchCase("common_understood_no", "phase4_common_understood", "noch nicht", "no", "phase4_common_understood"),
    BranchCase("common_feel_learning", "phase4_common_feel_after_learning", "traurig", "ready", "phase4_common_first_drag"),
    BranchCase("common_feel_aversion", "phase4_common_feel_after_aversion", "eklig", "ready", "phase4_common_collect_moments"),
    BranchCase("common_done", "phase4_common_done_signal", "sessel", "ready", "session_phase5_future"),
]


FLOW_SCENARIOS = [
    "phase1_anchor_core",
    "phase2_core",
    "common_step_core",
    "phase4_to_session_end",
    "phase4_origin_group_path",
    "phase4_origin_person_path",
    "person_switch_resolution_path",
]


def _all_semantic_node_ids() -> list[str]:
    node_ids: list[str] = []
    for node_id in sorted(available_node_ids()):
        if isinstance(get_node_spec(node_id), SemanticNodeSpec):
            node_ids.append(node_id)
    return node_ids


def _default_runtime_slots_for_node(node_id: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    if node_id.startswith("origin_"):
        slots["trigger_focus_ref"] = "die Gruppe"
    if node_id.startswith("group_") or node_id.startswith("person_switch_"):
        slots["named_person"] = "Peter"
    if node_id.startswith("group_"):
        slots.setdefault("trigger_focus_ref", "die Gruppe")
    if node_id in {"group_person_trigger_role", "group_person_trigger_core"}:
        slots["group_person_trigger_reason"] = "er lacht mich aus"
    if node_id == "group_person_trigger_core":
        slots["group_person_trigger_role"] = "von allen"
    return slots


def _call_model_only(
    client: Any,
    model: str,
    node_id: str,
    customer_message: str,
    *,
    runtime_slots: dict[str, str] | None = None,
    session_context: str = "",
    live_api_budget: LiveApiCallBudget | None = None,
) -> tuple[dict[str, Any], Any]:
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise ValueError(f"{node_id} is not semantic")

    payload = build_request(
        node_id,
        customer_message,
        clarify_attempt=0,
        session_context=session_context,
    )
    rendered_question = _render_runtime_question(node_id, runtime_slots or {})
    if rendered_question:
        payload["runtime_question"] = rendered_question
    if runtime_slots:
        payload["runtime_slots"] = dict(runtime_slots)

    messages = [
        {
            "role": "system",
            "content": (
                spec.system_prompt
                + (
                    f"\n\nKonkrete Laufzeitfrage fuer diesen Durchlauf:\n\"{rendered_question}\""
                    if rendered_question
                    else ""
                )
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    if live_api_budget is not None:
        live_api_budget.consume(node_id)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    parsed = parse_json_object(completion.choices[0].message.content or "")
    parsed = repair_semantic_payload(node_id, parsed)
    decision = validate_semantic_decision(node_id, parsed)
    return parsed, decision


def _wrong_input_for_node(node_id: str, spec: SemanticNodeSpec) -> str:
    if "provided_scale" in spec.allowed_intents:
        return "banane"
    if "yes" in spec.allowed_intents and "no" in spec.allowed_intents:
        return "kartoffel"
    if "whole_group" in spec.allowed_intents:
        return "banane"
    if "ready" in spec.allowed_intents:
        return "banane"
    return "banane"


def _question_input_for_node(node_id: str) -> str:
    return "wie meinst du das?"


def _support_input_for_node(node_id: str) -> str:
    return "es ist mir gerade zu viel"


def _abort_input_for_node(node_id: str) -> str:
    return "abbrechen"


def _run_branch_cases(
    client: Any,
    model: str,
    live_api_budget: LiveApiCallBudget,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for case in BRANCH_CASES:
        try:
            parsed, decision = _call_model_only(
                client,
                model,
                case.node_id,
                case.customer_message,
                runtime_slots=case.runtime_slots,
                session_context=case.session_context,
                live_api_budget=live_api_budget,
            )
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    suite="branch_cases",
                    label=case.label,
                    node_id=case.node_id,
                    customer_message=case.customer_message,
                    expected={"intent": case.expected_intent, "next_node": case.expected_next_node},
                    actual={"error": repr(exc)},
                )
            )
            continue
        if decision.intent != case.expected_intent or decision.next_node != case.expected_next_node:
            failures.append(
                ValidationFailure(
                    suite="branch_cases",
                    label=case.label,
                    node_id=case.node_id,
                    customer_message=case.customer_message,
                    expected={"intent": case.expected_intent, "next_node": case.expected_next_node},
                    actual=parsed,
                )
            )
    return failures


def _run_question_sweep(
    client: Any,
    model: str,
    live_api_budget: LiveApiCallBudget,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for node_id in _all_semantic_node_ids():
        try:
            parsed, decision = _call_model_only(
                client,
                model,
                node_id,
                _question_input_for_node(node_id),
                runtime_slots=_default_runtime_slots_for_node(node_id),
                live_api_budget=live_api_budget,
            )
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    suite="question_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_question_input_for_node(node_id),
                    expected={"intent": "question", "action": "answer_question", "next_node": node_id},
                    actual={"error": repr(exc)},
                )
            )
            continue
        if decision.intent != "question" or decision.action != "answer_question" or decision.next_node != node_id:
            failures.append(
                ValidationFailure(
                    suite="question_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_question_input_for_node(node_id),
                    expected={"intent": "question", "action": "answer_question", "next_node": node_id},
                    actual=parsed,
                )
            )
    return failures


def _run_abort_sweep(
    client: Any,
    model: str,
    live_api_budget: LiveApiCallBudget,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for node_id in _all_semantic_node_ids():
        spec = get_node_spec(node_id)
        assert isinstance(spec, SemanticNodeSpec)
        expected_next = spec.routing_rules["abort"]["next_node"]
        try:
            parsed, decision = _call_model_only(
                client,
                model,
                node_id,
                _abort_input_for_node(node_id),
                runtime_slots=_default_runtime_slots_for_node(node_id),
                live_api_budget=live_api_budget,
            )
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    suite="abort_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_abort_input_for_node(node_id),
                    expected={"intent": "abort", "action": "abort", "next_node": expected_next},
                    actual={"error": repr(exc)},
                )
            )
            continue
        if decision.intent != "abort" or decision.action != "abort" or decision.next_node != expected_next:
            failures.append(
                ValidationFailure(
                    suite="abort_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_abort_input_for_node(node_id),
                    expected={"intent": "abort", "action": "abort", "next_node": expected_next},
                    actual=parsed,
                )
            )
    return failures


def _run_support_sweep(
    client: Any,
    model: str,
    live_api_budget: LiveApiCallBudget,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for node_id in _all_semantic_node_ids():
        try:
            parsed, decision = _call_model_only(
                client,
                model,
                node_id,
                _support_input_for_node(node_id),
                runtime_slots=_default_runtime_slots_for_node(node_id),
                live_api_budget=live_api_budget,
            )
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    suite="support_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_support_input_for_node(node_id),
                    expected={"intent": "support_needed", "action": "support", "next_node": node_id},
                    actual={"error": repr(exc)},
                )
            )
            continue
        if decision.intent != "support_needed" or decision.action != "support" or decision.next_node != node_id:
            failures.append(
                ValidationFailure(
                    suite="support_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=_support_input_for_node(node_id),
                    expected={"intent": "support_needed", "action": "support", "next_node": node_id},
                    actual=parsed,
                )
            )
    return failures


def _run_invalid_sweep(
    client: Any,
    model: str,
    live_api_budget: LiveApiCallBudget,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for node_id in _all_semantic_node_ids():
        spec = get_node_spec(node_id)
        assert isinstance(spec, SemanticNodeSpec)
        bad_input = _wrong_input_for_node(node_id, spec)
        try:
            parsed, decision = _call_model_only(
                client,
                model,
                node_id,
                bad_input,
                runtime_slots=_default_runtime_slots_for_node(node_id),
                live_api_budget=live_api_budget,
            )
        except Exception as exc:
            failures.append(
                ValidationFailure(
                    suite="invalid_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=bad_input,
                    expected={"same_node": True, "safe_action": True},
                    actual={"error": repr(exc)},
                )
            )
            continue
        safe_action = decision.action in {"clarify", "support", "repeat", "answer_question"}
        if not safe_action:
            failures.append(
                ValidationFailure(
                    suite="invalid_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message=bad_input,
                    expected={"safe_action": True},
                    actual=parsed,
                )
            )
    return failures


def _run_silence_sweep() -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for node_id in _all_semantic_node_ids():
        runtime_slots = _default_runtime_slots_for_node(node_id)
        finished = False
        for attempt in range(0, 6):
            decision, reply = _handle_silence(node_id, attempt, runtime_slots)
            if decision is None:
                finished = True
                break
            if decision.next_node != node_id:
                finished = True
                break
            if not reply.strip():
                failures.append(
                    ValidationFailure(
                        suite="silence_sweep",
                        label=node_id,
                        node_id=node_id,
                        customer_message="",
                        expected={"non_empty_reply_or_transition": True},
                        actual={"attempt": attempt, "reply": reply, "next_node": decision.next_node},
                    )
                )
                finished = True
                break
        if not finished:
            failures.append(
                ValidationFailure(
                    suite="silence_sweep",
                    label=node_id,
                    node_id=node_id,
                    customer_message="",
                    expected={"terminated_within_attempts": 6},
                    actual={"attempts_checked": 6},
                )
            )
    return failures


def _run_flow_scenarios(live_api_budget: LiveApiCallBudget) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    for scenario_name in FLOW_SCENARIOS:
        if scenario_name not in SCENARIOS:
            failures.append(
                ValidationFailure(
                    suite="flow_scenarios",
                    label=scenario_name,
                    node_id=scenario_name,
                    customer_message="",
                    expected={"scenario_exists": True},
                    actual={"scenario_exists": False},
                )
            )
            continue
        code = run_batch(
            scenario_name,
            debug_model=False,
            semantic_provider="ft",
            live_api_budget=live_api_budget,
        )
        if code != 0:
            failures.append(
                ValidationFailure(
                    suite="flow_scenarios",
                    label=scenario_name,
                    node_id=scenario_name,
                    customer_message="",
                    expected={"exit_code": 0},
                    actual={"exit_code": code},
                )
            )
    return failures


def _coverage_warnings() -> list[str]:
    semantic_nodes = set(_all_semantic_node_ids())
    covered_nodes = {case.node_id for case in BRANCH_CASES}
    missing = sorted(semantic_nodes - covered_nodes)
    warnings: list[str] = []
    if missing:
        warnings.append(f"Branch cases missing semantic nodes: {', '.join(missing)}")
    return warnings


def estimate_live_api_calls() -> int:
    semantic_node_count = len(_all_semantic_node_ids())
    flow_scenario_calls = sum(len(SCENARIOS.get(name) or []) for name in FLOW_SCENARIOS)
    return (
        len(BRANCH_CASES)
        + semantic_node_count
        + semantic_node_count
        + semantic_node_count
        + semantic_node_count
        + flow_scenario_calls
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk validation matrix for semantic routing.")
    parser.add_argument("--live-api", action="store_true", help="Fuehrt den Validation-Lauf wirklich gegen OpenAI aus.")
    parser.add_argument("--max-api-calls", type=int, help="Verpflichtendes hartes Limit fuer Live-OpenAI-Calls.")
    parser.add_argument(
        "--approval-file",
        default=str(DEFAULT_APPROVAL_FILE),
        help="Pfad zur Repo-Freigabedatei fuer Live-OpenAI.",
    )
    args = parser.parse_args()

    estimated_calls = estimate_live_api_calls()
    if not args.live_api:
        print("Live-OpenAI ist fuer dieses Skript standardmaessig blockiert.")
        print(f"Geschaetzte Live-Calls fuer einen Voll-Lauf: {estimated_calls}")
        print(
            "Zum bewussten Live-Lauf braucht es: OPENAI_LIVE_API_ALLOWED=1, "
            "eine gueltige backend/live_api_approval.json und --live-api --max-api-calls <n>."
        )
        return 0

    live_api_budget = build_live_api_budget(
        "run_session_validation_matrix.py",
        estimated_calls=estimated_calls,
        requested_max_calls=args.max_api_calls,
        approval_file=args.approval_file,
    )
    print(live_api_budget.summary())

    client, model = resolve_openai_client()
    report_path = Path(r"C:\Projekte\test_app\backend\session_validation_report.json")

    failures: list[ValidationFailure] = []
    failures.extend(_run_branch_cases(client, model, live_api_budget))
    failures.extend(_run_question_sweep(client, model, live_api_budget))
    failures.extend(_run_abort_sweep(client, model, live_api_budget))
    failures.extend(_run_support_sweep(client, model, live_api_budget))
    failures.extend(_run_invalid_sweep(client, model, live_api_budget))
    failures.extend(_run_silence_sweep())
    failures.extend(_run_flow_scenarios(live_api_budget))

    warnings = _coverage_warnings()
    report = {
        "semantic_node_count": len(_all_semantic_node_ids()),
        "branch_case_count": len(BRANCH_CASES),
        "flow_scenarios": FLOW_SCENARIOS,
        "estimated_live_api_calls": estimated_calls,
        "allowed_live_api_calls": live_api_budget.allowed_calls,
        "consumed_live_api_calls": live_api_budget.consumed_calls,
        "warnings": warnings,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validation report written to: {report_path}")
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"FAILURES: {len(failures)}")
    for failure in failures:
        print(
            f"[{failure.suite}] {failure.label} | node={failure.node_id} | "
            f"input={failure.customer_message!r} | expected={failure.expected} | actual={failure.actual}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
