import unittest

from phase4_semantic_prompt_prototype import (
    ScriptNodeSpec,
    available_node_ids,
    build_request,
    get_node_spec,
    get_semantic_node_spec,
    repair_semantic_payload,
    script_reply_for_decision,
    validate_semantic_decision,
)


class SemanticPromptPrototypeTests(unittest.TestCase):
    def test_build_request_uses_new_branch_node(self) -> None:
        payload = build_request(
            "hell_feel_branch",
            "das fuehlt sich angenehm an",
            clarify_attempt=2,
            session_context="heller Zustand",
        )
        self.assertEqual(payload["node_id"], "hell_feel_branch")
        self.assertEqual(payload["clarify_attempt"], 2)
        self.assertEqual(payload["customer_message"], "das fuehlt sich angenehm an")
        self.assertEqual(payload["session_context"], "heller Zustand")

    def test_validate_decision_routes_pleasant_to_hypnose_loch(self) -> None:
        decision = validate_semantic_decision(
            "hell_feel_branch",
            {
                "intent": "pleasant",
                "action": "transition",
                "next_node": "hell_hypnose_loch_intro",
                "confidence": 0.94,
                "reason": "Das Helle wird klar angenehm beschrieben.",
            },
        )
        self.assertEqual(decision.next_node, "hell_hypnose_loch_intro")

    def test_validate_decision_routes_unpleasant_to_regulation(self) -> None:
        decision = validate_semantic_decision(
            "hell_feel_branch",
            {
                "intent": "unpleasant",
                "action": "transition",
                "next_node": "hell_regulation_choice",
                "confidence": 0.9,
                "reason": "Das Helle wird als drueckend beschrieben.",
            },
        )
        self.assertEqual(decision.next_node, "hell_regulation_choice")

    def test_validate_decision_keeps_question_in_same_node(self) -> None:
        decision = validate_semantic_decision(
            "hell_feel_branch",
            {
                "intent": "question",
                "action": "answer_question",
                "next_node": "hell_feel_branch",
                "confidence": 0.88,
                "reason": "Der Kunde stellt eine Rueckfrage.",
            },
        )
        self.assertEqual(decision.next_node, "hell_feel_branch")
        self.assertEqual(script_reply_for_decision("hell_feel_branch", decision), "Achte hier nur darauf, wie sich dieses Helle fuer dich im Erleben anfuehlt. Ist es fuer dich eher angenehm oder eher unangenehm?")

    def test_validate_decision_routes_known_to_backtrace(self) -> None:
        decision = validate_semantic_decision(
            "dark_known_branch",
            {
                "intent": "known",
                "action": "transition",
                "next_node": "dark_backtrace_terminal",
                "confidence": 0.91,
                "reason": "Der Kunde beschreibt das Gefuehl als bereits bekannt.",
            },
        )
        self.assertEqual(decision.next_node, "dark_backtrace_terminal")

    def test_validate_decision_routes_new_to_origin(self) -> None:
        decision = validate_semantic_decision(
            "dark_known_branch",
            {
                "intent": "new",
                "action": "transition",
                "next_node": "dark_origin_terminal",
                "confidence": 0.89,
                "reason": "Der Kunde beschreibt dies als ersten Ursprung.",
            },
        )
        self.assertEqual(decision.next_node, "dark_origin_terminal")

    def test_dark_origin_terminal_contains_common_origin_intro(self) -> None:
        spec = get_node_spec("dark_origin_terminal")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertIn("Gut. Dann bleiben wir jetzt in genau dieser Szene", spec.script_text)

    def test_origin_self_resolution_intro_now_flows_into_need_clarification(self) -> None:
        spec = get_node_spec("origin_self_resolution_intro")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.next_node, "origin_self_need")

    def test_origin_self_need_routes_into_release_intro(self) -> None:
        decision = validate_semantic_decision(
            "origin_self_need",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_self_release_intro",
                "confidence": 0.93,
                "reason": "Das fehlende innere Beduerfnis wurde benannt.",
            },
        )
        self.assertEqual(decision.next_node, "origin_self_release_intro")

    def test_dark_scene_people_who_now_stays_in_search_phase_and_routes_to_age(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_people_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 0.9,
                "reason": "Die Personen in der Suchszene wurden benannt.",
            },
        )
        self.assertEqual(decision.next_node, "dark_scene_age")

    def test_dark_scene_happening_now_starts_origin_resolution_and_routes_to_trigger_source(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_happening",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_source",
                "confidence": 0.9,
                "reason": "Das Ereignis in der Ursprungsszene wurde beschrieben.",
            },
        )
        self.assertEqual(decision.next_node, "origin_trigger_source")

    def test_dark_scene_people_who_clarify_reply_is_no_longer_command_like(self) -> None:
        spec = get_semantic_node_spec("dark_scene_people_who")
        clarify_reply = spec.same_node_replies["unclear"]
        self.assertNotIn("Schau einfach, wen du dort genau wahrnimmst.", clarify_reply)
        self.assertIn("wer oder was dort fuer dich wahrnehmbar wird", clarify_reply)

    def test_origin_trigger_source_now_routes_into_known_vs_new_check(self) -> None:
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 0.9,
                "reason": "Der staerkste Ausloeser in der Ursprungsszene wurde benannt.",
            },
        )
        self.assertEqual(decision.next_node, "origin_trigger_known_branch")

    def test_origin_trigger_known_branch_can_route_back_to_earlier_origin(self) -> None:
        decision = validate_semantic_decision(
            "origin_trigger_known_branch",
            {
                "intent": "known",
                "action": "transition",
                "next_node": "dark_backtrace_terminal",
                "confidence": 0.9,
                "reason": "Das Gefuehl ist bereits aus frueheren Momenten bekannt.",
            },
        )
        self.assertEqual(decision.next_node, "dark_backtrace_terminal")

    def test_origin_trigger_known_branch_can_continue_directly_into_origin_cause_owner(self) -> None:
        decision = validate_semantic_decision(
            "origin_trigger_known_branch",
            {
                "intent": "new",
                "action": "transition",
                "next_node": "origin_cause_owner",
                "confidence": 0.9,
                "reason": "Das Gefuehl zeigt sich hier neu und deshalb wird genau in dieser Szene weitergearbeitet.",
            },
        )
        self.assertEqual(decision.next_node, "origin_cause_owner")

    def test_origin_scene_relevance_can_continue_with_current_scene(self) -> None:
        decision = validate_semantic_decision(
            "origin_scene_relevance",
            {
                "intent": "resolve_here",
                "action": "transition",
                "next_node": "origin_cause_owner",
                "confidence": 0.9,
                "reason": "Die Szene soll hier weiter bearbeitet werden.",
            },
        )
        self.assertEqual(decision.next_node, "origin_cause_owner")

    def test_origin_scene_relevance_can_route_back_to_earlier_origin(self) -> None:
        decision = validate_semantic_decision(
            "origin_scene_relevance",
            {
                "intent": "older_origin",
                "action": "transition",
                "next_node": "dark_backtrace_terminal",
                "confidence": 0.9,
                "reason": "Die Szene verweist noch auf einen frueheren Ursprung.",
            },
        )
        self.assertEqual(decision.next_node, "dark_backtrace_terminal")

    def test_dark_feeling_intensity_can_skip_repeated_feeling_question(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_feeling_intensity",
            {
                "intent": "feeling_and_intensity",
                "action": "transition",
                "next_node": "dark_known_branch",
                "confidence": 0.92,
                "reason": "Die Antwort enthaelt bereits ein konkretes unmittelbares Gefuehl.",
            },
        )
        self.assertEqual(decision.next_node, "dark_known_branch")

    def test_dark_feeling_intensity_can_still_ask_immediate_feeling_when_needed(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_feeling_intensity",
            {
                "intent": "intensity_only",
                "action": "transition",
                "next_node": "dark_scene_immediate_feeling",
                "confidence": 0.88,
                "reason": "Die Antwort beschreibt vor allem die Staerke.",
            },
        )
        self.assertEqual(decision.next_node, "dark_scene_immediate_feeling")

    def test_repair_payload_fills_missing_metadata(self) -> None:
        repaired = repair_semantic_payload(
            "hell_feel_branch",
            {
                "intent": "pleasant",
                "action": "transition",
                "next_node": "hell_hypnose_loch_intro",
            },
        )
        self.assertIn("confidence", repaired)
        self.assertIn("reason", repaired)

    def test_repair_payload_recovers_intent_from_action_and_next_node(self) -> None:
        repaired = repair_semantic_payload(
            "hell_feel_branch",
            {
                "intent": "answer_question",
                "action": "answer_question",
                "next_node": "hell_feel_branch",
            },
        )
        self.assertEqual(repaired["intent"], "question")

    def test_repair_payload_canonicalizes_route_from_valid_intent(self) -> None:
        repaired = repair_semantic_payload(
            "hell_regulation_choice",
            {
                "intent": "question",
                "action": "clarify",
                "next_node": "hell_regulation_check",
            },
        )
        self.assertEqual(repaired["intent"], "question")
        self.assertEqual(repaired["action"], "answer_question")
        self.assertEqual(repaired["next_node"], "hell_regulation_choice")

    def test_script_node_contains_exact_hypnose_loch_text(self) -> None:
        spec = get_node_spec("hell_hypnose_loch_intro")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertIn("Das ist haeufig ein Hypnose-Loch", spec.script_text)
        self.assertEqual(spec.next_node, "hell_hypnose_wait")

    def test_resolved_script_now_flows_to_dark_known_branch(self) -> None:
        spec = get_node_spec("hell_post_resolved_terminal")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.next_node, "dark_known_branch")

    def test_available_node_ids_include_new_branch_nodes(self) -> None:
        self.assertIn("hell_light_level", available_node_ids())
        self.assertIn("hell_feel_branch", available_node_ids())
        self.assertIn("hell_hypnose_loch_intro", available_node_ids())
        self.assertIn("hell_regulation_choice", available_node_ids())
        self.assertIn("dark_known_branch", available_node_ids())
        self.assertIn("dark_backtrace_terminal", available_node_ids())
        self.assertIn("dark_origin_terminal", available_node_ids())
        self.assertIn("origin_trigger_known_branch", available_node_ids())
        self.assertIn("origin_scene_relevance", available_node_ids())
        self.assertIn("group_branch_intro", available_node_ids())
        self.assertIn("group_source_kind", available_node_ids())
        self.assertIn("person_switch_self_understands", available_node_ids())

    def test_regulation_choice_has_entry_script(self) -> None:
        spec = get_semantic_node_spec("hell_regulation_choice")
        self.assertIn("Licht jetzt etwas weicher", spec.entry_script)

    def test_dark_known_question_stays_in_same_node(self) -> None:
        decision = validate_semantic_decision(
            "dark_known_branch",
            {
                "intent": "question",
                "action": "answer_question",
                "next_node": "dark_known_branch",
                "confidence": 0.84,
                "reason": "Der Kunde braucht eine Einordnungshilfe.",
            },
        )
        self.assertEqual(decision.next_node, "dark_known_branch")
        self.assertEqual(
            script_reply_for_decision("dark_known_branch", decision),
            "Hier geht es nicht darum, das Gefuehl genauer zu erklaeren. Es reicht, wenn du kurz einordnest, ob es dir aus frueheren Momenten schon bekannt vorkommt oder ob es sich hier zum ersten Mal zeigt.",
        )

    def test_group_whole_group_routes_to_scope_question(self) -> None:
        decision = validate_semantic_decision(
            "group_source_kind",
            {
                "intent": "whole_group",
                "action": "transition",
                "next_node": "group_whole_scope",
                "confidence": 0.91,
                "reason": "Der Kunde beschreibt die ganze Gruppe als Ausloeser.",
            },
        )
        self.assertEqual(decision.next_node, "group_whole_scope")

    def test_group_specific_person_routes_to_specific_person_intro(self) -> None:
        decision = validate_semantic_decision(
            "group_source_kind",
            {
                "intent": "one_person",
                "action": "transition",
                "next_node": "group_specific_person_intro",
                "confidence": 0.93,
                "reason": "Der Kunde grenzt den Ausloeser auf eine Person aus der Gruppe ein.",
            },
        )
        self.assertEqual(decision.next_node, "group_specific_person_intro")

    def test_group_switch_can_finish_after_understanding(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_self_understands",
            {
                "intent": "yes",
                "action": "transition",
                "next_node": "group_resolution_complete",
                "confidence": 0.88,
                "reason": "Das neue Verstaendnis hilft bereits weiter.",
            },
        )
        self.assertEqual(decision.next_node, "group_resolution_complete")

    def test_system_prompt_prioritizes_question_and_support_meta_signals(self) -> None:
        spec = get_semantic_node_spec("group_person_trigger_reason")
        self.assertIn("wie meinst du das", spec.system_prompt)
        self.assertIn("es ist mir zu viel", spec.system_prompt)
        self.assertIn("du verstehst mich nicht", spec.system_prompt)
        self.assertIn("ich schlafe fast ein", spec.system_prompt)
        self.assertIn("eigentlich habe ich keine Lust mehr".lower(), spec.system_prompt.lower())

    def test_system_prompt_marks_nothing_yet_and_nonsense_explicitly(self) -> None:
        spec = get_semantic_node_spec("scene_access_followup")
        self.assertIn("nothing_yet", spec.system_prompt)
        self.assertIn("Nonsense", spec.system_prompt)

    def test_dark_scene_perception_system_prompt_distinguishes_missing_visual_access_from_nothing(self) -> None:
        spec = get_semantic_node_spec("dark_scene_perception")
        self.assertIn("ich sehe nichts", spec.system_prompt)
        self.assertIn("fehlenden visuellen Zugang", spec.system_prompt)

    def test_system_prompt_allows_explanatory_answers_with_leading_yes(self) -> None:
        spec = get_semantic_node_spec("group_person_trigger_core")
        self.assertIn("Einleitungswort wie 'ja'", spec.system_prompt)

    def test_group_person_trigger_core_ready_meaning_accepts_brief_status_replies(self) -> None:
        spec = get_semantic_node_spec("group_person_trigger_core")
        self.assertIn("'ja'", spec.intent_meanings["ready"])
        self.assertIn("'noch nicht'", spec.intent_meanings["ready"])

    def test_dark_scene_feeling_intensity_question_uses_softer_therapeutic_wording(self) -> None:
        spec = get_semantic_node_spec("dark_scene_feeling_intensity")
        self.assertIn("Wie zeigt sich dieses ungute Gefuehl dort gerade?", spec.question_text)
        self.assertIn("wie wuerdest du es in diesem Moment beschreiben", spec.question_text)

    def test_dark_known_question_uses_therapeutic_known_vs_origin_wording(self) -> None:
        spec = get_semantic_node_spec("dark_known_branch")
        self.assertIn("Wenn du dieses Gefuehl jetzt so wahrnimmst", spec.question_text)
        self.assertIn("Kommt es dir aus frueheren Momenten schon bekannt vor", spec.question_text)
        self.assertIn("zeigt es sich hier zum ersten Mal", spec.question_text)

    def test_origin_trigger_known_question_uses_more_spoken_wording(self) -> None:
        spec = get_semantic_node_spec("origin_trigger_known_branch")
        self.assertIn("Wenn du jetzt bei diesem Gefuehl bleibst", spec.question_text)
        self.assertIn("Kommt es dir aus frueheren Momenten schon bekannt vor", spec.question_text)
        self.assertIn("zeigt es sich hier in dieser Szene zum ersten Mal", spec.question_text)


if __name__ == "__main__":
    unittest.main()
