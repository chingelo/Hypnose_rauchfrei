import io
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import run_session_sandbox as sandbox
from run_session_sandbox import (
    CATEGORY_CHOICE_NODES,
    NAMED_PERSON_INPUT_NODES,
    QUESTION_ANSWER_HINTS,
    STRICT_PHASE4_YES_NO_NODES,
    _answer_question_in_context,
    _build_local_intent_prompt,
    _capture_runtime_slots,
    _classify_focus_reference,
    _contextual_same_node_reply,
    _diagnostic_empty_input_reply,
    _local_intent_payload_to_decision,
    _render_runtime_question,
    _route_runtime_next_node,
    _should_use_contextual_same_node_reply,
    _dynamic_same_node_reply,
    _decision_for_empty_input,
    _empty_input_reply,
    _extract_named_person_label,
    _extract_scale_value,
    _handle_silence,
    _looks_like_nonanswer_noise,
    _is_reengagement_signal,
    _is_question_announcement,
    _max_silence_attempts_before_outro,
    _local_router_predecision,
    _local_session_decision,
    _prepend_silence_to_pcm,
    _print_paged_block,
    _prepare_tts_text,
    _question_announcement_reply,
    _render_runtime_text,
    _restore_german_umlauts_for_tts,
    _sanitize_customer_facing_answer,
    _scale_confirmation_prefix,
    _silence_timeout_seconds,
    _tts_post_block_pause_ms,
    _tts_lead_in_ms,
    _tts_provider,
    run_interactive,
    call_semantic_node,
)
from session_sandbox_orchestrator import (
    PHASE2_SCENE_GUIDANCE_SCRIPT,
    PHASE4_COMMON_FIRST_CIGARETTE_CONSEQUENCES,
    ScriptNodeSpec,
    available_node_ids,
    build_request,
    get_node_spec,
    render_script_node,
    repair_semantic_payload,
    script_reply_for_decision,
    validate_semantic_decision,
)

GENERIC_PERSON_STATUS_REPLIES = (
    "ja ich erkenne die person",
    "ich erkenne die person",
    "ja ich sehe die person",
    "ich seh die person",
    "ich kenne die person",
    "ich weiss wer es ist",
    "ich weiss wer sie ist",
    "ich weiss wer die person ist",
    "ja die person steht vor mir",
    "die person steht vor mir",
    "die person ist da",
    "die person ist hier",
    "die person ist klar",
    "die person ist jetzt klar",
    "die person ist jetzt da",
    "eine person steht vor mir",
)

PERSON_VISIBILITY_READY_REPLIES = (
    "ich sehe ihn",
    "er steht vor mir",
    "die person steht vor mir",
    "ich weiss wer die person ist",
    "die person ist jetzt klar",
)

AUDIO_CONTACT_READY_REPLIES = (
    "ich hoere dich",
    "ich habe dich gehoert",
    "habs gehoert",
)

IMPACT_VISIBILITY_READY_REPLIES = (
    "ich sehe was passiert",
    "ich sehe die wirkung",
    "ich sehe es jetzt",
)

HEARD_CUSTOMER_READY_REPLIES = (
    "ich habe es gehoert",
    "habs gehoert",
    "ich hoere was gesagt wurde",
)


class SessionSandboxOrchestratorTests(unittest.TestCase):
    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = SessionSandboxOrchestratorTests._FakeMessage(content)

    class _FakeCompletion:
        def __init__(self, content: str) -> None:
            self.choices = [SessionSandboxOrchestratorTests._FakeChoice(content)]

    class _FakeCompletions:
        def __init__(self, response_text: str) -> None:
            self.response_text = response_text
            self.last_kwargs: dict | None = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return SessionSandboxOrchestratorTests._FakeCompletion(self.response_text)

    class _FakeChat:
        def __init__(self, response_text: str) -> None:
            self.completions = SessionSandboxOrchestratorTests._FakeCompletions(response_text)

    class _FakeClient:
        def __init__(self, response_text: str) -> None:
            self.chat = SessionSandboxOrchestratorTests._FakeChat(response_text)

    class _FakeLocalIntentRouter:
        def __init__(self, response_text: str = '{"intent":"unclear"}') -> None:
            self.response_text = response_text
            self.calls = 0
            self.last_prompt: str | None = None

        def infer_intent(self, prompt: str) -> str:
            self.calls += 1
            self.last_prompt = prompt
            return self.response_text

    def test_phase1_script_flows_to_preflight(self) -> None:
        spec = get_node_spec("session_phase1_intro")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.next_node, "session_phase1_preflight_check")

    def test_phase1_preflight_continue_routes_forward(self) -> None:
        decision = validate_semantic_decision(
            "session_phase1_preflight_check",
            {
                "intent": "continue",
                "action": "transition",
                "next_node": "session_phase1_setup_script",
                "confidence": 0.9,
                "reason": "Der Kunde signalisiert, dass alles bereit ist.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase1_setup_script")

    def test_phase1_setup_script_no_longer_mentions_flight_mode(self) -> None:
        spec = get_node_spec("session_phase1_setup_script")
        self.assertIsInstance(spec, ScriptNodeSpec)
        assert isinstance(spec, ScriptNodeSpec)
        self.assertNotIn("Flugmodus", spec.script_text)

    def test_phase1_anchor_handles_technical_issue_in_place(self) -> None:
        decision = validate_semantic_decision(
            "session_phase1_anchor_after_setup",
            {
                "intent": "technical_issue",
                "action": "support",
                "next_node": "session_phase1_anchor_after_setup",
                "confidence": 0.88,
                "reason": "Der Kunde beschreibt ein technisches oder aeusseres Thema.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase1_anchor_after_setup")
        self.assertIn("Ton", script_reply_for_decision("session_phase1_anchor_after_setup", decision))

    def test_yes_no_node_routes_yes_forward(self) -> None:
        decision = validate_semantic_decision(
            "session_phase2_ready",
            {
                "intent": "yes",
                "action": "transition",
                "next_node": "session_phase2_post_ready_script",
                "confidence": 0.91,
                "reason": "Der Kunde bestaetigt die Bereitschaft klar.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase2_post_ready_script")

    def test_yes_no_node_keeps_no_on_same_node(self) -> None:
        decision = validate_semantic_decision(
            "session_phase2_ready",
            {
                "intent": "no",
                "action": "support",
                "next_node": "session_phase2_ready",
                "confidence": 0.88,
                "reason": "Der Kunde ist noch nicht bereit.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase2_ready")
        self.assertIn("Sobald du bereit bist", script_reply_for_decision("session_phase2_ready", decision))

    def test_scale_node_routes_number_forward(self) -> None:
        decision = validate_semantic_decision(
            "session_phase2_scale_before",
            {
                "intent": "provided_scale",
                "action": "transition",
                "next_node": "session_phase2_post_scale_before_script",
                "confidence": 0.94,
                "reason": "Der Kunde nennt einen klaren Skalenwert.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase2_post_scale_before_script")

    def test_phase4_external_terminal_is_overridden_into_origin_event_block(self) -> None:
        script_text, next_node = render_script_node("dark_origin_terminal")
        self.assertIn("wichtiger Ursprung", script_text)
        self.assertEqual(next_node, "dark_scene_happening")

    def test_phase4_backtrace_terminal_loops_back_to_light_dark_question(self) -> None:
        script_text, next_node = render_script_node("dark_backtrace_terminal")
        self.assertNotIn("frueheren Punkt angekommen", script_text)
        self.assertTrue(script_text.strip().endswith("Null."))
        self.assertEqual(next_node, "hell_light_level")

    def test_phase4_post_countdown_entry_flows_into_light_question(self) -> None:
        script_text, next_node = render_script_node("session_phase4_post_countdown_entry")
        self.assertIn("Am Ursprung", script_text)
        self.assertEqual(next_node, "hell_light_level")

    def test_hell_light_level_dark_route_enters_v2_dark_scene_block(self) -> None:
        decision = validate_semantic_decision(
            "hell_light_level",
            {
                "intent": "darker_or_other",
                "action": "transition",
                "next_node": "dark_scene_perception",
                "confidence": 0.95,
                "reason": "Die Szene wird als dunkel beschrieben.",
            },
        )
        self.assertEqual(decision.next_node, "dark_scene_perception")

    def test_common_ready_node_routes_forward(self) -> None:
        decision = validate_semantic_decision(
            "phase4_common_done_signal",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "session_phase5_future",
                "confidence": 0.86,
                "reason": "Der Kunde signalisiert, dass er wieder im Sessel angekommen ist.",
            },
        )
        self.assertEqual(decision.next_node, "session_phase5_future")

    def test_build_request_preserves_context(self) -> None:
        payload = build_request(
            "phase4_common_feel_after_learning",
            "das fuehlt sich klar an",
            clarify_attempt=1,
            session_context="erste aversion",
        )
        self.assertEqual(payload["node_id"], "phase4_common_feel_after_learning")
        self.assertEqual(payload["clarify_attempt"], 1)
        self.assertEqual(payload["session_context"], "erste aversion")

    def test_repair_payload_infers_session_intent(self) -> None:
        repaired = repair_semantic_payload(
            "phase4_common_done_signal",
            {
                "action": "transition",
                "next_node": "session_phase5_future",
            },
        )
        self.assertEqual(repaired["intent"], "ready")
        self.assertIn("confidence", repaired)
        self.assertIn("reason", repaired)

    def test_repair_payload_canonicalizes_session_route_from_valid_intent(self) -> None:
        repaired = repair_semantic_payload(
            "session_phase2_ready",
            {
                "intent": "question",
                "action": "clarify",
                "next_node": "session_phase2_post_ready_script",
            },
        )
        self.assertEqual(repaired["intent"], "question")
        self.assertEqual(repaired["action"], "answer_question")
        self.assertEqual(repaired["next_node"], "session_phase2_ready")

    def test_empty_input_falls_back_to_unclear(self) -> None:
        decision = _decision_for_empty_input("session_phase1_anchor_after_setup")
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "session_phase1_anchor_after_setup")

    def test_optional_intro_anchor_auto_continues_on_silence(self) -> None:
        decision, reply = _handle_silence("session_phase1_anchor_after_setup", 0)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "continue")
        self.assertEqual(decision.next_node, "session_phase1_mindset_script")
        self.assertIn("gehen wir jetzt weiter", reply)

    def test_required_question_stays_in_place_on_silence(self) -> None:
        decision, reply = _handle_silence("session_phase2_ready", 0)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "session_phase2_ready")
        self.assertIn("Ich frage dich noch einmal etwas klarer", reply)
        self.assertIn("Ja oder Nein", reply)

    def test_required_question_second_silence_uses_specific_followup(self) -> None:
        decision, reply = _handle_silence("session_phase2_ready", 1)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "session_phase2_ready")
        self.assertIn("Ja oder Nein", reply)
        self.assertNotIn("Ich warte noch einen Augenblick", reply)

    def test_group_representative_name_first_silence_rephrases_question(self) -> None:
        reply = _empty_input_reply("group_representative_name", 0)
        self.assertIn("namentlich nennen", reply)
        self.assertNotIn("Nimm dir kurz einen Moment", reply)

    def test_group_representative_name_second_silence_can_fall_back_to_waiting_prompt(self) -> None:
        reply = _empty_input_reply("group_representative_name", 1)
        self.assertIn("Sobald dir die passende Person klarer wird", reply)

    def test_build_local_intent_prompt_contains_allowed_intents(self) -> None:
        prompt = _build_local_intent_prompt(
            "dark_known_branch",
            "zum ersten mal",
            session_context="",
            runtime_slots={},
        )
        self.assertIn('"task": "routing_intent"', prompt)
        self.assertIn('"allowed_intents"', prompt)
        self.assertIn('"question_text"', prompt)
        self.assertIn('"node_goal"', prompt)
        self.assertIn('"dark_known_branch"', prompt)

    def test_local_intent_payload_maps_to_canonical_route(self) -> None:
        parsed, decision = _local_intent_payload_to_decision(
            "dark_known_branch",
            {"intent": "new"},
        )
        self.assertEqual(parsed["intent"], "new")
        self.assertEqual(parsed["action"], "transition")
        self.assertEqual(parsed["next_node"], "dark_origin_terminal")
        self.assertEqual(decision.next_node, "dark_origin_terminal")

    def test_local_router_predecision_does_not_consume_free_ready_content(self) -> None:
        decision = _local_router_predecision("dark_scene_age", "12")
        self.assertIsNone(decision)

    def test_local_router_predecision_handles_explicit_dark_for_hell_light_level(self) -> None:
        decision = _local_router_predecision("hell_light_level", "dunkel")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "darker_or_other")
        self.assertEqual(decision.next_node, "dark_scene_perception")

    def test_local_session_decision_leaves_origin_trigger_known_new_state_for_model(self) -> None:
        decision = _local_session_decision("origin_trigger_known_branch", "hier ist es neu")
        self.assertIsNone(decision)

    def test_local_session_decision_leaves_origin_trigger_known_known_state_for_model(self) -> None:
        decision = _local_session_decision("origin_trigger_known_branch", "das kenne ich schon")
        self.assertIsNone(decision)

    def test_local_session_decision_leaves_origin_trigger_source_concrete_focus_for_model(self) -> None:
        decision = _local_session_decision("origin_trigger_source", "die person vermutlich")
        self.assertIsNone(decision)

    def test_local_session_decision_leaves_origin_trigger_source_generic_placeholder_for_model(self) -> None:
        decision = _local_session_decision("origin_trigger_source", "ich sehe etwas")
        self.assertIsNone(decision)

    def test_local_router_predecision_blocks_acknowledgement_only_reply_on_immediate_feeling_node(self) -> None:
        decision = _local_router_predecision("dark_scene_immediate_feeling", "ok")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_scene_immediate_feeling")

    def test_local_router_predecision_blocks_acknowledgement_only_reply_on_identity_node(self) -> None:
        decision = _local_router_predecision("dark_scene_people_who", "ok")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_scene_people_who")

    def test_local_session_decision_handles_late_dark_answer_on_scene_access_followup(self) -> None:
        decision = _local_session_decision("scene_access_followup", "es ist dunkel")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "visual_dark")
        self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_session_decision_handles_late_hell_answer_on_scene_access_followup(self) -> None:
        decision = _local_session_decision("scene_access_followup", "es ist hell")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "visual_hell")
        self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_session_decision_maps_nonvisual_scene_access_followup_inputs(self) -> None:
        cases = [
            ("ich hoere etwas", "nonvisual_access", "scene_access_body_bridge_intro"),
            ("ich spuere druck", "nonvisual_access", "scene_access_body_bridge_intro"),
            ("ich rieche rauch", "nonvisual_access", "scene_access_body_bridge_intro"),
            ("nichts", "nothing_yet", "scene_access_body_bridge_intro"),
        ]
        for phrase, expected_intent, expected_next in cases:
            with self.subTest(phrase=phrase):
                decision = _local_session_decision("scene_access_followup", phrase)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, expected_intent)
                self.assertEqual(decision.next_node, expected_next)

    def test_local_session_decision_handles_visual_fragments_on_scene_access_followup(self) -> None:
        cases = ["ein gebaeude", "blau", "schulhof"]
        for phrase in cases:
            with self.subTest(phrase=phrase):
                decision = _local_session_decision("scene_access_followup", phrase)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "visual_hell")
                self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_session_decision_reclassifies_dark_on_hell_feel_branch(self) -> None:
        decision = _local_session_decision("hell_feel_branch", "es ist dunkel")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "reclassified_dark")
        self.assertEqual(decision.next_node, "dark_scene_perception")

    def test_local_session_decision_reclassifies_dark_typo_without_space_on_hell_regulation_choice(self) -> None:
        decision = _local_session_decision("hell_regulation_choice", "es istdunkel")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "reclassified_dark")
        self.assertEqual(decision.next_node, "dark_scene_perception")

    def test_call_semantic_node_local_intent_uses_hell_light_guardrail_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"hell_light"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "hell_light_level",
            "dunkel",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "darker_or_other")
        self.assertEqual(decision.next_node, "dark_scene_perception")

    def test_scale_question_first_silence_rephrases_question(self) -> None:
        reply = _empty_input_reply("session_phase2_scale_before", 0)
        self.assertIn("Welche Zahl zwischen 1 und 10", reply)

    def test_scene_found_first_silence_rephrases_with_craving_context(self) -> None:
        reply = _empty_input_reply("session_phase2_scene_found", 0)
        self.assertIn("Verlangen nach der Zigarette", reply)
        self.assertIn("Suchtdruck", reply)

    def test_hypnose_wait_first_silence_rephrases_wait_options(self) -> None:
        reply = _empty_input_reply("hell_hypnose_wait", 0)
        self.assertIn("Loest es sich noch auf", reply)
        self.assertIn("bereits aufgeloest", reply)

    def test_hell_light_level_first_silence_transitions_to_scene_access_followup(self) -> None:
        decision, reply = _handle_silence("hell_light_level", 0)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "scene_access_followup")
        self.assertEqual(reply, "")

    def test_scene_access_followup_first_silence_uses_v2_hint_text(self) -> None:
        reply = _empty_input_reply("scene_access_followup", 0)
        self.assertIn("Falls du gerade nichts erkennen kannst", reply)
        self.assertIn("besonders wenn wir weit zurueckgehen", reply)
        self.assertIn("erster Eindruck", reply)
        self.assertNotIn("Antworte einfach dann", reply)

    def test_scene_access_followup_empty_input_varies_across_attempts(self) -> None:
        first = _empty_input_reply("scene_access_followup", 0)
        second = _empty_input_reply("scene_access_followup", 1)
        third = _empty_input_reply("scene_access_followup", 2)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertNotEqual(first, third)

    def test_scene_access_followup_question_opens_other_senses(self) -> None:
        spec = get_node_spec("scene_access_followup")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("hoeren", spec.question_text)
        self.assertIn("riechen", spec.question_text)
        self.assertIn("schmecken", spec.question_text)
        self.assertIn("bestimmtes Gefuehl", spec.question_text)

    def test_dark_scene_perception_first_silence_stays_in_place_with_soft_followup(self) -> None:
        decision, reply = _handle_silence("dark_scene_perception", 0)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "dark_scene_perception")
        self.assertIn("klarer", reply)
        self.assertNotEqual(reply, get_node_spec("dark_scene_perception").question_text)

    def test_dark_scene_mode_clarify_first_silence_does_not_repeat_question_verbatim(self) -> None:
        spec = get_node_spec("dark_scene_mode_clarify")
        assert not isinstance(spec, ScriptNodeSpec)
        reply = _empty_input_reply("dark_scene_mode_clarify", 0)
        self.assertNotEqual(reply, spec.question_text)
        self.assertIn("einordnen", reply)

    def test_scene_access_followup_second_silence_transitions_to_body_bridge(self) -> None:
        decision, reply = _handle_silence("scene_access_followup", 1)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "scene_access_body_bridge_intro")
        self.assertEqual(reply, "")

    def test_scene_access_body_bridge_flows_into_nonvisual_question(self) -> None:
        script_text, next_node = render_script_node("scene_access_body_bridge_intro")
        self.assertIn("ohne klares Bild", script_text)
        self.assertEqual(next_node, "dark_scene_other_sense")

    def test_phase2_silence_schedule_uses_early_phase_policy(self) -> None:
        self.assertEqual(_silence_timeout_seconds("session_phase2_ready", 0), 10.0)
        self.assertEqual(_silence_timeout_seconds("session_phase2_ready", 1), 14.0)
        self.assertEqual(_silence_timeout_seconds("session_phase2_ready", 3), 18.0)

    def test_phase4_silence_schedule_uses_late_phase_policy(self) -> None:
        self.assertEqual(_silence_timeout_seconds("hell_light_level", 0), 12.0)
        self.assertEqual(_silence_timeout_seconds("hell_light_level", 1), 16.0)
        self.assertEqual(_silence_timeout_seconds("hell_light_level", 4), 15.0)

    def test_exploratory_scene_nodes_allow_more_silence_before_outro(self) -> None:
        self.assertEqual(
            _max_silence_attempts_before_outro("dark_scene_people_who"),
            sandbox.MAX_SILENCE_ATTEMPTS_BEFORE_OUTRO + 2,
        )
        self.assertEqual(
            _max_silence_attempts_before_outro("session_phase2_ready"),
            sandbox.MAX_SILENCE_ATTEMPTS_BEFORE_OUTRO,
        )

    def test_required_question_fourth_silence_uses_diagnostic_prompt(self) -> None:
        reply = _empty_input_reply("session_phase2_ready", 5)
        self.assertIn("kurze klare Rueckmeldung", reply)
        self.assertNotIn("beende ich die Sitzung", reply)

    def test_required_question_fifth_silence_uses_inactivity_warning(self) -> None:
        decision, reply = _handle_silence("session_phase2_ready", 6)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "session_phase2_ready")
        self.assertIn("zusammenhaengenden Rahmen fortsetzen", reply)
        self.assertIn("sicher und stimmig weiter", reply)
        self.assertIn("ruhigen Ausleitung", reply)

    def test_required_question_sixth_silence_uses_final_warning(self) -> None:
        decision, reply = _handle_silence("session_phase2_ready", 7)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "session_phase2_ready")
        self.assertIn("noch immer keine Rueckmeldung", reply)
        self.assertIn("ruhig aus der Sitzung heraus", reply)
        self.assertIn("an einem sicheren Punkt", reply)

    def test_required_question_seventh_silence_starts_safe_outro(self) -> None:
        decision, reply = _handle_silence("session_phase2_ready", 8)
        self.assertIsNone(decision)
        self.assertIn("beginne ich jetzt eine ruhige Ausleitung", reply)

    def test_exploratory_scene_node_fifth_silence_does_not_warn_yet(self) -> None:
        decision, reply = _handle_silence("dark_scene_people_who", 4)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "dark_scene_people_who")
        self.assertNotIn("zusammenhaengenden Rahmen fortsetzen", reply)
        self.assertNotIn("noch immer keine Rueckmeldung", reply)

    def test_exploratory_scene_node_seventh_silence_uses_inactivity_warning(self) -> None:
        decision, reply = _handle_silence("dark_scene_people_who", 8)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.next_node, "dark_scene_people_who")
        self.assertIn("zusammenhaengenden Rahmen fortsetzen", reply)

    def test_scene_access_diagnostic_prompt_mentions_no_access_and_other_senses(self) -> None:
        reply = _empty_input_reply("dark_scene_who", 7)
        self.assertIn("noch nichts richtig greifbar", reply)
        self.assertIn("ueber den Koerper, einen Geruch, einen Geschmack oder eine Temperatur", reply)
        self.assertNotIn("nicht tief genug", reply)

    def test_dark_scene_people_who_empty_input_reply_uses_open_wording(self) -> None:
        reply = _empty_input_reply("dark_scene_people_who", 1)
        self.assertIn("wer oder was dort fuer dich", reply)
        self.assertNotIn("Person oder Gruppe", reply)

    def test_scene_access_followup_visual_access_goes_directly_to_visual_detail(self) -> None:
        decision = validate_semantic_decision(
            "scene_access_followup",
            {
                "intent": "visual_hell",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Sobald ein Bild auftaucht, wird direkt geklaert, was konkret gesehen wird.",
            },
        )
        self.assertEqual(decision.next_node, "dark_scene_who")

    def test_route_runtime_next_node_reclassifies_late_dark_answer_on_scene_access_followup(self) -> None:
        decision = validate_semantic_decision(
            "scene_access_followup",
            {
                "intent": "visual_dark",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Spaete Dunkel-Einordnung erkannt.",
            },
        )
        routed = _route_runtime_next_node("scene_access_followup", decision, {}, "es ist dunkel")
        self.assertEqual(routed, "dark_scene_perception")

    def test_route_runtime_next_node_reclassifies_late_hell_answer_on_scene_access_followup(self) -> None:
        decision = validate_semantic_decision(
            "scene_access_followup",
            {
                "intent": "visual_hell",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Spaete Hell-Einordnung erkannt.",
            },
        )
        routed = _route_runtime_next_node("scene_access_followup", decision, {}, "es ist hell")
        self.assertEqual(routed, "hell_feel_branch")

    def test_route_runtime_next_node_reclassifies_late_dark_answer_on_body_bridge(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaete Dunkel-Einordnung erkannt.",
            },
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, {}, "dunkel")
        self.assertEqual(routed, "dark_scene_perception")

    def test_route_runtime_next_node_reclassifies_late_visual_access_on_body_bridge(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter visueller Zugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "ich sehe was",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "ich sehe was")
        self.assertEqual(routed, "dark_scene_who")

    def test_route_runtime_next_node_reclassifies_visual_fragment_on_body_bridge(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter visueller Fragmentzugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "blau",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "blau")
        self.assertEqual(routed, "dark_scene_who")

    def test_route_runtime_next_node_skips_visual_detail_question_when_body_bridge_already_contains_specific_visual_fragment(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter visueller Fragmentzugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "ein gebaeude",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "ein gebaeude")
        self.assertEqual(routed, "dark_scene_age")

    def test_route_runtime_next_node_skips_visual_detail_question_when_body_bridge_already_contains_specific_people_scene(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter visueller Zugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "ich sehe eine person",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "ich sehe eine person")
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_sends_generic_person_fragment_on_body_bridge_directly_to_people_followup(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter visueller Zugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "eine person",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "eine person")
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_reclassifies_late_audio_access_on_body_bridge(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_other_sense",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_first_spuerbar",
                "confidence": 1.0,
                "reason": "Spaeter auditiver Zugang erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_other_sense",
            "ich hoere was",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_other_sense", decision, runtime_slots, "ich hoere was")
        self.assertEqual(routed, "dark_scene_audio_detail")

    def test_call_semantic_node_local_intent_handles_late_dark_answer_on_scene_access_followup_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"nonvisual_access"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "scene_access_followup",
            "es ist dunkel",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "visual_dark")

    def test_call_semantic_node_local_intent_handles_late_dark_answer_on_body_bridge_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"unclear"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_scene_other_sense",
            "dunkel",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "ready")

    def test_call_semantic_node_local_intent_handles_late_visual_access_on_body_bridge_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"unclear"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_scene_other_sense",
            "ich sehe was",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "ready")

    def test_dark_scene_who_question_is_open_and_not_person_only(self) -> None:
        spec = get_node_spec("dark_scene_who")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Was siehst du dort genau?")
        reply = _empty_input_reply("dark_scene_who", 0)
        self.assertNotEqual(reply, spec.question_text)
        self.assertIn("sichtbar", reply)

    def test_dark_scene_perception_question_matches_v2_wording(self) -> None:
        spec = get_node_spec("dark_scene_perception")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Und was nimmst du dort sonst noch wahr, siehst du oder hoerst du was?")

    def test_dark_scene_mode_clarify_question_matches_v2_wording(self) -> None:
        spec = get_node_spec("dark_scene_mode_clarify")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Okay, was kannst du wahrnehmen? Siehst du jemand oder hoerst du was?")

    def test_dark_scene_audio_detail_question_matches_v2_wording(self) -> None:
        spec = get_node_spec("dark_scene_audio_detail")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Was hoerst du dort genau?")

    def test_dark_scene_other_sense_question_matches_v2_wording(self) -> None:
        spec = get_node_spec("dark_scene_other_sense")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(
            spec.question_text,
            "Wenn du dort nichts klar siehst oder hoerst: Was nimmst du ueber Koerper, Geruch, Geschmack oder Temperatur wahr?",
        )

    def test_dark_scene_first_spuerbar_question_matches_v2_wording(self) -> None:
        spec = get_node_spec("dark_scene_first_spuerbar")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Und was ist dort als Erstes am deutlichsten spuerbar?")

    def test_dark_scene_immediate_feeling_question_uses_open_non_example_wording(self) -> None:
        spec = get_node_spec("dark_scene_immediate_feeling")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(
            spec.question_text,
            "Wenn du da jetzt direkt hineinspuerst: Wie zeigt sich dieses Gefuehl in diesem Moment ganz unmittelbar?",
        )

    def test_ready_nodes_system_prompt_allows_short_content_answers(self) -> None:
        spec = get_node_spec("dark_scene_immediate_feeling")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("sehr kurze, aber inhaltlich passende Antworten", spec.system_prompt)
        self.assertIn("'druck'", spec.system_prompt)

    def test_perception_nodes_system_prompt_distinguishes_audio_and_other_sense_cues(self) -> None:
        spec = get_node_spec("dark_scene_perception")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("lachen", spec.system_prompt)
        self.assertIn("audio", spec.system_prompt)
        self.assertIn("other_sense", spec.system_prompt)

    def test_group_source_kind_system_prompt_accepts_single_name_as_one_person(self) -> None:
        spec = get_node_spec("group_source_kind")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("Peter", spec.system_prompt)
        self.assertIn("one_person", spec.system_prompt)

    def test_dark_scene_immediate_feeling_ready_meaning_accepts_single_word_feelings(self) -> None:
        spec = get_node_spec("dark_scene_immediate_feeling")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("'druck'", spec.intent_meanings["ready"])
        self.assertIn("'angst'", spec.intent_meanings["ready"])

    def test_group_person_trigger_core_ready_meaning_accepts_short_abstract_causes(self) -> None:
        spec = get_node_spec("group_person_trigger_core")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("'dazugehoeren'", spec.intent_meanings["ready"])
        self.assertIn("'gruppendruck'", spec.intent_meanings["ready"])

    def test_person_switch_why_ready_meaning_accepts_short_abstract_explanations(self) -> None:
        spec = get_node_spec("person_switch_why")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("'ueberforderung'", spec.intent_meanings["ready"])
        self.assertIn("'druck'", spec.intent_meanings["ready"])

    def test_dark_scene_people_who_question_uses_open_non_assumptive_wording(self) -> None:
        spec = get_node_spec("dark_scene_people_who")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.question_text, "Kannst du mir sagen, wer oder was dort fuer dich erkennbar wird?")

    def test_dark_scene_people_who_question_hint_uses_open_non_assumptive_wording(self) -> None:
        hint = QUESTION_ANSWER_HINTS["dark_scene_people_who"]
        self.assertIn("wer oder was", hint)
        self.assertNotIn("welche Person oder Personen", hint)

    def test_group_scope_first_silence_rephrases_branch_choice(self) -> None:
        reply = _empty_input_reply("group_whole_scope", 0)
        self.assertIn("stellvertretende Person", reply)
        self.assertIn("mehrere Personen", reply)

    def test_group_next_person_check_first_silence_rephrases_yes_no_question(self) -> None:
        reply = _empty_input_reply("group_next_person_check", 0)
        self.assertIn("weitere Person", reply)
        self.assertIn("Ja oder Nein", reply)

    def test_origin_cause_owner_first_silence_is_contextualized_and_not_generic_waiting(self) -> None:
        reply = _empty_input_reply(
            "origin_cause_owner",
            0,
            {"trigger_focus_ref": "die Gruppe"},
        )
        self.assertIn("diese Gruppe", reply)
        self.assertIn("etwas in dir selbst", reply)
        self.assertNotIn("Antworte einfach dann", reply)

    def test_origin_cause_owner_second_silence_uses_next_staged_reply(self) -> None:
        reply = _empty_input_reply(
            "origin_cause_owner",
            1,
            {"trigger_focus_ref": "die Gruppe"},
        )
        self.assertIn("Sobald du es einordnen kannst", reply)
        self.assertIn("diese Gruppe", reply)

    def test_generic_ready_followup_no_longer_uses_old_waiting_formula(self) -> None:
        reply = _empty_input_reply("phase4_common_done_signal", 1)
        self.assertIn("erster Eindruck", reply)
        self.assertNotIn("Antworte einfach dann", reply)
        self.assertNotIn("Ich warte noch einen Augenblick", reply)

    def test_diagnostic_scene_access_reply_is_soft_and_therapeutic(self) -> None:
        reply = _diagnostic_empty_input_reply("dark_scene_perception", get_node_spec("dark_scene_perception"), {})
        self.assertIn("ist das in Ordnung", reply)
        self.assertIn("Bleib einfach noch einen Moment dabei", reply)
        self.assertIn("ueber den Koerper", reply)
        self.assertNotIn("Ich brauche jetzt nur noch", reply)

    def test_local_parser_maps_ok_to_yes_for_phase2_ready(self) -> None:
        decision = _local_session_decision("session_phase2_ready", "ok")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "yes")

    def test_local_parser_does_not_block_explanatory_answer_with_leading_yes_on_group_person_trigger_core(self) -> None:
        decision = _local_session_decision(
            "group_person_trigger_core",
            "ja sie lachen weil ich der einzige bin der nicht raucht",
        )
        self.assertIsNone(decision)

    def test_local_parser_accepts_bare_yes_as_ready_on_group_person_trigger_core(self) -> None:
        decision = _local_session_decision("group_person_trigger_core", "ja")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "person_switch_ready_intro")

    def test_local_parser_accepts_not_clear_status_as_ready_on_group_person_trigger_core(self) -> None:
        decision = _local_session_decision("group_person_trigger_core", "noch nicht")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "person_switch_ready_intro")

    def test_local_parser_keeps_bare_yes_unclear_on_group_person_trigger_reason(self) -> None:
        decision = _local_session_decision("group_person_trigger_reason", "ja")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_local_parser_maps_question_for_scene_found(self) -> None:
        decision = _local_session_decision("session_phase2_scene_found", "verstehe ich nicht")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "session_phase2_scene_found")

    def test_local_meta_state_maps_not_understood_complaint_to_question(self) -> None:
        decision = _local_session_decision("dark_scene_feeling_intensity", "du verstehst mich wohl nicht")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "dark_scene_feeling_intensity")

    def test_local_meta_state_maps_fatigue_to_support_needed(self) -> None:
        decision = _local_session_decision("dark_known_branch", "ich schlafe fast ein")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "dark_known_branch")

    def test_local_meta_state_maps_ambivalence_to_support_needed(self) -> None:
        decision = _local_session_decision("group_person_ready", "ich weiss nicht ob ich das will")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "group_person_ready")

    def test_local_meta_state_maps_hostile_reply_to_support_needed(self) -> None:
        decision = _local_session_decision("group_person_trigger_reason", "du bist ein idiot")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_local_semantics_map_visual_negative_only_to_other_sense(self) -> None:
        decision = _local_session_decision("dark_scene_perception", "ich sehe nichts")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "other_sense")
        self.assertEqual(decision.next_node, "dark_scene_other_sense")

    def test_local_semantics_map_not_currently_perceivable_feeling_to_support(self) -> None:
        decision = _local_session_decision("dark_scene_feeling_intensity", "ich nehme es gerade nicht mehr wahr")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "dark_scene_feeling_intensity")

    def test_local_semantics_map_explicit_single_person_choice_in_group_source_kind(self) -> None:
        decision = _local_session_decision("group_source_kind", "eine person")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "one_person")
        self.assertEqual(decision.next_node, "group_specific_person_intro")

    def test_local_semantics_keep_mixed_origin_owner_reply_in_same_node(self) -> None:
        decision = _local_session_decision("origin_cause_owner", "von beiden vermutlich")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "origin_cause_owner")

    def test_local_semantics_map_explicit_other_party_on_origin_cause_owner(self) -> None:
        for text in ("die person", "diese person", "er", "sie", "jemand anderes"):
            with self.subTest(text=text):
                decision = _local_session_decision("origin_cause_owner", text)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "someone_else")
                self.assertEqual(decision.next_node, "origin_other_target_kind")

    def test_local_semantics_map_external_focus_families_on_origin_cause_owner(self) -> None:
        for text in ("mein freund", "mein vater", "die gruppe", "der geruch", "die zigarette"):
            with self.subTest(text=text):
                decision = _local_session_decision("origin_cause_owner", text)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "someone_else")
                self.assertEqual(decision.next_node, "origin_other_target_kind")

    def test_local_semantics_map_explicit_self_on_origin_cause_owner(self) -> None:
        for text in ("in mir", "von mir", "etwas in mir", "bei mir selbst"):
            with self.subTest(text=text):
                decision = _local_session_decision("origin_cause_owner", text)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "self")
                self.assertEqual(decision.next_node, "origin_self_resolution_intro")

    def test_local_semantics_keep_weiss_es_nicht_in_same_origin_cause_owner_node(self) -> None:
        decision = _local_session_decision("origin_cause_owner", "weiss es nicht")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "origin_cause_owner")

    def test_local_semantics_map_origin_scene_relevance_to_resolve_here(self) -> None:
        for text in (
            "das muessen wir hier loesen",
            "das ist genau hier der kern",
            "das steht fuer meinen vater",
            "das sollten wir hier erst anschauen",
        ):
            with self.subTest(text=text):
                decision = _local_session_decision("origin_scene_relevance", text)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "resolve_here")
                self.assertEqual(decision.next_node, "origin_cause_owner")

    def test_local_semantics_map_origin_scene_relevance_to_older_origin(self) -> None:
        for text in (
            "das fuehrt mich noch weiter zurueck",
            "das ist noch nicht der ursprung",
            "da kommt noch etwas frueheres davor",
        ):
            with self.subTest(text=text):
                decision = _local_session_decision("origin_scene_relevance", text)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "older_origin")
                self.assertEqual(decision.next_node, "dark_backtrace_terminal")

    def test_local_semantics_leave_known_vs_new_content_for_model(self) -> None:
        cases = (
            ("dark_known_branch", "das kenne ich schon"),
            ("dark_known_branch", "zum ersten mal"),
            ("origin_trigger_known_branch", "das kenne ich schon"),
            ("origin_trigger_known_branch", "zum ersten mal"),
            ("dark_known_branch", "erste mal"),
            ("origin_trigger_known_branch", "erste mal"),
        )
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                decision = _local_session_decision(node_id, phrase)
                self.assertIsNone(decision)

    def test_local_semantics_map_explicit_person_choice_for_origin_other_target_kind(self) -> None:
        decision = _local_session_decision("origin_other_target_kind", "eine person")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "person")
        self.assertEqual(decision.next_node, "origin_person_name")

    def test_local_semantics_rejects_too_short_category_reply_for_origin_other_target_kind(self) -> None:
        decision = _local_session_decision("origin_other_target_kind", "e")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "origin_other_target_kind")

    def test_local_semantics_leave_negated_known_replies_for_model(self) -> None:
        cases = (
            ("dark_known_branch", "ich kenne es nicht"),
            ("dark_known_branch", "ich kenne es noch nciht"),
            ("origin_trigger_known_branch", "ich kenne es noch nciht"),
        )
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                decision = _local_session_decision(node_id, phrase)
                self.assertIsNone(decision)

    def test_local_semantics_leave_feeling_intensity_content_for_model(self) -> None:
        for phrase in ("wie ein druck in der brust, sehr stark", "stark", "druck in der brust", "enge in der brust"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(_local_session_decision("dark_scene_feeling_intensity", phrase, restrict_scope=False))

    def test_local_semantics_map_hell_light_level_non_light_answer_to_unclear(self) -> None:
        decision = _local_session_decision("hell_light_level", "ich bin am gleichen ort wie vorher")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "hell_light_level")

    def test_contextual_same_node_reply_explains_same_scene_report_on_hell_light_level(self) -> None:
        decision = validate_semantic_decision(
            "hell_light_level",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "hell_light_level",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        reply = _contextual_same_node_reply(
            None,
            "",
            "hell_light_level",
            decision,
            customer_message="ich bin am gleichen ort wie vorher",
            clarify_attempt=0,
            session_context="",
            runtime_slots={},
            silence=False,
        )
        self.assertIn("gleichen oder an einem sehr aehnlichen Moment", reply)
        self.assertIn("eher hell, eher dunkel oder gemischt", reply)

    def test_local_semantics_map_dark_feeling_intensity_acute_distress_to_support_needed(self) -> None:
        decision = _local_session_decision("dark_scene_feeling_intensity", "ich kann fast nicht atmen")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "dark_scene_feeling_intensity")

    def test_local_semantics_map_dark_immediate_feeling_acute_distress_to_support_needed(self) -> None:
        decision = _local_session_decision("dark_scene_immediate_feeling", "mir bleibt die luft weg")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "dark_scene_immediate_feeling")

    def test_local_semantics_treat_question_on_group_multiple_people_name_as_question(self) -> None:
        decision = _local_session_decision("group_multiple_people_name", "mit was beginnen?")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "group_multiple_people_name")

    def test_extract_named_person_label_rejects_direct_question(self) -> None:
        self.assertIsNone(_extract_named_person_label("mit was beginnen?"))

    def test_extract_named_person_label_rejects_generic_person_placeholder_with_article(self) -> None:
        self.assertIsNone(_extract_named_person_label("die person"))

    def test_extract_named_person_label_rejects_hedged_generic_person_placeholder(self) -> None:
        self.assertIsNone(_extract_named_person_label("vermutlich die person"))
        self.assertIsNone(_extract_named_person_label("vermutlich die person dort"))
        self.assertIsNone(_extract_named_person_label("wahrscheinlich eine person"))
        self.assertIsNone(_extract_named_person_label("die person vermutlich"))
        self.assertIsNone(_extract_named_person_label("die person vielleicht"))
        self.assertIsNone(_extract_named_person_label("die person dort"))
        self.assertIsNone(_extract_named_person_label("die person selber"))

    def test_extract_named_person_label_rejects_hedged_specific_person_labels_instead_of_faking_names(self) -> None:
        for phrase in ("wahrscheinlich fritz", "fritz vermutlich", "mein vater vermutlich", "vielleicht mein vater"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(_extract_named_person_label(phrase))

    def test_extract_named_person_label_rejects_generic_visual_placeholders(self) -> None:
        self.assertIsNone(_extract_named_person_label("ich sehe etwas"))
        self.assertIsNone(_extract_named_person_label("ich sehe jemanden"))

    def test_named_person_input_nodes_never_accept_direct_question_as_name(self) -> None:
        for node_id in NAMED_PERSON_INPUT_NODES:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "mit was beginnen?")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "question")
                self.assertEqual(decision.next_node, node_id)

    def test_named_person_input_nodes_reject_generic_person_placeholder(self) -> None:
        for node_id in NAMED_PERSON_INPUT_NODES:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "eine person")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, node_id)

    def test_named_person_input_nodes_accept_real_named_person(self) -> None:
        for node_id in NAMED_PERSON_INPUT_NODES:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "Peter")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "ready")

    def test_named_person_input_nodes_reject_group_like_reference(self) -> None:
        for node_id in NAMED_PERSON_INPUT_NODES:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "die anderen da hinten")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, node_id)

    def test_named_person_input_nodes_map_global_meta_states_before_content(self) -> None:
        samples = {
            "question": "wie meinst du das",
            "support_needed": "ich brauche kurz noch einen moment",
            "fatigue": "ich schlafe fast ein",
            "hostile": "du bist ein idiot",
            "abort": "abbrechen",
        }
        for node_id in NAMED_PERSON_INPUT_NODES:
            with self.subTest(node_id=node_id, case="question"):
                decision = _local_session_decision(node_id, samples["question"])
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "question")
                self.assertEqual(decision.next_node, node_id)
            for case_name in ("support_needed", "fatigue", "hostile"):
                with self.subTest(node_id=node_id, case=case_name):
                    decision = _local_session_decision(node_id, samples[case_name])
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "support_needed")
                    self.assertEqual(decision.next_node, node_id)
            with self.subTest(node_id=node_id, case="abort"):
                decision = _local_session_decision(node_id, samples["abort"])
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "abort")
                self.assertEqual(decision.next_node, "abort_confirmation")

    def test_category_choice_nodes_map_global_meta_states_before_content(self) -> None:
        for node_id in CATEGORY_CHOICE_NODES:
            with self.subTest(node_id=node_id, case="question"):
                decision = _local_session_decision(node_id, "wie meinst du das")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "question")
                self.assertEqual(decision.next_node, node_id)
            for case_name, sample in (
                ("ambivalence", "ich weiss nicht ob ich das will"),
                ("fatigue", "ich schlafe fast ein"),
                ("hostile", "du bist ein idiot"),
            ):
                with self.subTest(node_id=node_id, case=case_name):
                    decision = _local_session_decision(node_id, sample)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "support_needed")
                    self.assertEqual(decision.next_node, node_id)
            with self.subTest(node_id=node_id, case="abort"):
                decision = _local_session_decision(node_id, "abbrechen")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "abort")
                self.assertEqual(decision.next_node, "abort_confirmation")

    def test_yes_no_nodes_map_global_meta_states_before_yes_no(self) -> None:
        for node_id in STRICT_PHASE4_YES_NO_NODES:
            with self.subTest(node_id=node_id, case="question"):
                decision = _local_session_decision(node_id, "wie meinst du das")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "question")
                self.assertEqual(decision.next_node, node_id)
            for case_name, sample in (
                ("ambivalence", "ich weiss nicht ob ich das will"),
                ("fatigue", "ich schlafe fast ein"),
                ("hostile", "du bist ein idiot"),
            ):
                with self.subTest(node_id=node_id, case=case_name):
                    decision = _local_session_decision(node_id, sample)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "support_needed")
                    self.assertEqual(decision.next_node, node_id)
            with self.subTest(node_id=node_id, case="abort"):
                decision = _local_session_decision(node_id, "abbrechen")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "abort")
                self.assertEqual(decision.next_node, "abort_confirmation")

    def test_local_parser_does_not_treat_unklar_as_klar(self) -> None:
        decision = _local_session_decision("session_phase2_scale_clear", "das ist mir noch unklar")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "session_phase2_scale_clear")

    def test_question_announcement_is_detected(self) -> None:
        self.assertTrue(_is_question_announcement("ich habe eine frage"))
        self.assertTrue(_is_question_announcement("frage"))
        self.assertFalse(_is_question_announcement("was meinst du mit ungestoert"))

    def test_nonanswer_noise_helper_detects_obvious_noise(self) -> None:
        for phrase in ("asdf", "...", "??", "hm", "hmm", "kp", "lalala", "blabla", "ich bin ein toaster", "x"):
            with self.subTest(phrase=phrase):
                self.assertTrue(_looks_like_nonanswer_noise(phrase))

    def test_nonanswer_noise_helper_keeps_potentially_meaningful_content_open(self) -> None:
        for phrase in ("druck", "eine person", "stimmen", "wecker", "banane", "ich rieche rauch", "ich bin ein kind"):
            with self.subTest(phrase=phrase):
                self.assertFalse(_looks_like_nonanswer_noise(phrase))

    def test_question_announcement_reply_escalates(self) -> None:
        self.assertIn("Frage", _question_announcement_reply(0))
        self.assertIn("Frage selbst", _question_announcement_reply(1))

    def test_local_parser_maps_question_announcement_to_question(self) -> None:
        decision = _local_session_decision("session_phase1_anchor_after_setup", "ich habe eine frage")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "session_phase1_anchor_after_setup")

    def test_local_parser_maps_embedded_runtime_stop_question_to_question(self) -> None:
        decision = _local_session_decision("dark_scene_who", "menschen wieso stoppt es hier")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_parser_maps_noe_to_no(self) -> None:
        decision = _local_session_decision("session_phase2_ready", "noe")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "no")
        self.assertEqual(decision.next_node, "session_phase2_ready")

    def test_local_parser_maps_phase4_yes_no_node_to_no(self) -> None:
        decision = _local_session_decision("person_switch_aware_trigger", "nein")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "no")
        self.assertEqual(decision.next_node, "person_switch_aware_trigger")

    def test_local_parser_accepts_explicit_person_visibility_status_on_group_person_ready(self) -> None:
        decision = _local_session_decision("group_person_ready", "er steht vor mir")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "yes")
        self.assertEqual(decision.next_node, "group_person_handoff")

    def test_local_parser_does_not_misclassify_third_person_content_as_support_meta_state(self) -> None:
        decision = _local_session_decision("person_switch_heard_customer", "sie war ueberfordert")
        self.assertIsNone(decision)

    def test_local_parser_blocks_plain_yes_for_person_switch_why(self) -> None:
        decision = _local_session_decision("person_switch_why", "ja")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "person_switch_why")

    def test_local_parser_leaves_explanation_content_for_person_switch_why_to_model(self) -> None:
        decision = _local_session_decision("person_switch_why", "er war ueberfordert")
        self.assertIsNone(decision)

    def test_call_semantic_node_blocks_meta_and_placeholder_replies_on_explanation_nodes_without_router(self) -> None:
        cases = [
            ("group_person_trigger_reason", "ich verstehe"),
            ("group_person_trigger_reason", "ich sehe etwas"),
            ("group_person_trigger_role", "ich verstehe"),
            ("group_person_trigger_role", "ich sehe etwas"),
            ("group_person_trigger_core", "ich verstehe"),
            ("group_person_trigger_core", "ich sehe etwas"),
            ("person_switch_why", "ich verstehe"),
            ("person_switch_why", "ich sehe etwas"),
        ]
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    node_id,
                    phrase,
                    runtime_slots={"named_person": "diese Person"},
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 0)
                self.assertEqual(parsed["source"], "local_parser")
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, node_id)

    def test_call_semantic_node_keeps_group_person_trigger_core_short_status_ready_without_router(self) -> None:
        for phrase in ("ja", "noch nicht"):
            with self.subTest(phrase=phrase):
                fake_router = self._FakeLocalIntentRouter('{"intent":"unclear"}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    "group_person_trigger_core",
                    phrase,
                    runtime_slots={"named_person": "diese Person"},
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 0)
                self.assertEqual(parsed["source"], "local_parser")
                self.assertEqual(decision.intent, "ready")
                self.assertEqual(decision.next_node, "person_switch_ready_intro")

    def test_local_parser_detects_question_for_group_person_trigger_reason(self) -> None:
        decision = _local_session_decision("group_person_trigger_reason", "wie meinst du das")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_call_semantic_node_local_intent_keeps_question_in_same_node_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_person_trigger_reason",
            "wie meinst du das genau",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_call_semantic_node_local_intent_keeps_support_in_same_node_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_person_trigger_reason",
            "ich brauche kurz noch einen moment",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_call_semantic_node_local_intent_keeps_fatigue_meta_state_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"known"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_known_branch",
            "ich schlafe fast ein",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "dark_known_branch")

    def test_call_semantic_node_local_intent_keeps_ambivalence_meta_state_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"yes"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_person_ready",
            "ich weiss nicht ob ich das will",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "support_needed")
        self.assertEqual(decision.next_node, "group_person_ready")

    def test_call_semantic_node_local_intent_uses_router_for_explanation_content(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_person_trigger_reason",
            "er lacht mich vor allen aus",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 1)
        self.assertEqual(parsed["source"], "local_intent_router")
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "group_person_trigger_role")

    def test_call_semantic_node_local_intent_uses_router_for_structured_free_content(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_scene_age",
            "12",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 1)
        self.assertEqual(parsed["source"], "local_intent_router")
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "dark_scene_feeling_intensity")

    def test_call_semantic_node_local_intent_uses_router_for_dark_feeling_intensity_content(self) -> None:
        cases = (
            ("stark", "intensity_only", "dark_scene_immediate_feeling"),
            ("druck in der brust", "feeling_and_intensity", "dark_known_branch"),
        )
        for phrase, intent, next_node in cases:
            with self.subTest(phrase=phrase):
                fake_router = self._FakeLocalIntentRouter(f'{{"intent":"{intent}"}}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    "dark_scene_feeling_intensity",
                    phrase,
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 1)
                self.assertEqual(parsed["source"], "local_intent_router")
                self.assertEqual(decision.intent, intent)
                self.assertEqual(decision.next_node, next_node)

    def test_call_semantic_node_local_intent_uses_router_for_known_vs_new_content(self) -> None:
        cases = (
            ("dark_known_branch", "das kenne ich schon", "known", "dark_backtrace_terminal"),
            ("dark_known_branch", "zum ersten mal", "new", "dark_origin_terminal"),
            ("origin_trigger_known_branch", "das kenne ich schon", "known", "dark_backtrace_terminal"),
            ("origin_trigger_known_branch", "zum ersten mal", "new", "origin_cause_owner"),
        )
        for node_id, phrase, intent, next_node in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                fake_router = self._FakeLocalIntentRouter(f'{{"intent":"{intent}"}}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    node_id,
                    phrase,
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 1)
                self.assertEqual(parsed["source"], "local_intent_router")
                self.assertEqual(decision.intent, intent)
                self.assertEqual(decision.next_node, next_node)

    def test_call_semantic_node_local_intent_uses_router_for_representative_free_content_nodes(self) -> None:
        cases = (
            ("dark_scene_happening", "sie lachen mich aus", "origin_trigger_source"),
            ("origin_self_need", "schutz", "origin_self_release_intro"),
            ("group_person_trigger_role", "er steht fuer die gruppe", "group_person_trigger_core"),
            ("person_switch_why", "er will dazugehoeren", "person_switch_aware_trigger"),
        )
        for node_id, phrase, next_node in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    node_id,
                    phrase,
                    runtime_slots={"named_person": "Fritz"},
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 1)
                self.assertEqual(parsed["source"], "local_intent_router")
                self.assertEqual(decision.intent, "ready")
                self.assertEqual(decision.next_node, next_node)

    def test_call_semantic_node_trace_logs_local_session_decision(self) -> None:
        trace_events: list[dict[str, object]] = []

        parsed, decision = call_semantic_node(
            None,
            "",
            "group_person_ready",
            "ja",
            trace_logger=trace_events.append,
        )

        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "yes")
        self.assertTrue(any(event["stage"] == "call_start" for event in trace_events))
        self.assertTrue(any(event["stage"] == "local_session_decision_hit" for event in trace_events))
        self.assertIn("_trace", parsed)

    def test_call_semantic_node_trace_logs_local_intent_router_prompt_and_raw_response(self) -> None:
        trace_events: list[dict[str, object]] = []
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')

        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_scene_age",
            "12",
            local_intent_router=fake_router,
            trace_logger=trace_events.append,
        )

        self.assertEqual(parsed["source"], "local_intent_router")
        self.assertEqual(decision.intent, "ready")
        self.assertTrue(any(event["stage"] == "local_intent_prompt_built" for event in trace_events))
        self.assertTrue(any(event["stage"] == "local_intent_router_raw_response" for event in trace_events))
        self.assertTrue(any(event["stage"] == "local_intent_decision_validated" for event in trace_events))
        self.assertEqual(fake_router.calls, 1)

    def test_call_semantic_node_trace_logs_semantic_raw_response_and_validation(self) -> None:
        trace_events: list[dict[str, object]] = []
        fake_client = self._FakeClient(
            '{"intent":"ready","action":"transition","next_node":"dark_scene_age","confidence":0.9,"reason":"Konkreter visueller Inhalt erkannt."}'
        )

        parsed, decision = call_semantic_node(
            fake_client,
            "gpt-test",
            "dark_scene_who",
            "ein gebaeude",
            trace_logger=trace_events.append,
        )

        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "dark_scene_age")
        self.assertTrue(any(event["stage"] == "semantic_request_prepared" for event in trace_events))
        self.assertTrue(any(event["stage"] == "semantic_chat_completion_start" for event in trace_events))
        self.assertTrue(any(event["stage"] == "semantic_chat_completion_raw_response" for event in trace_events))
        self.assertTrue(any(event["stage"] == "semantic_payload_repaired" for event in trace_events))
        self.assertTrue(any(event["stage"] == "semantic_decision_validated" for event in trace_events))
        self.assertIn("_trace", parsed)

    def test_call_semantic_node_local_intent_uses_router_for_origin_trigger_source_free_content(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "origin_trigger_source",
            "die person raucht",
            runtime_slots={"named_person": "Fritz"},
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 1)
        self.assertEqual(parsed["source"], "local_intent_router")
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "origin_trigger_known_branch")

    def test_call_semantic_node_local_intent_blocks_origin_trigger_source_generic_placeholder_before_router(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "origin_trigger_source",
            "ich sehe etwas",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "origin_trigger_source")

    def test_call_semantic_node_local_intent_uses_group_scope_predecision_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"one_person"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_source_kind",
            "die ganze gruppe",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "whole_group")
        self.assertEqual(decision.next_node, "group_whole_scope")

    def test_call_semantic_node_local_intent_uses_explicit_single_person_group_scope_without_router_call(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"multiple_people"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_source_kind",
            "eine person",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "one_person")
        self.assertEqual(decision.next_node, "group_specific_person_intro")

    def test_call_semantic_node_local_intent_does_not_treat_question_as_person_name(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_multiple_people_name",
            "mit was beginnen?",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "question")
        self.assertEqual(decision.next_node, "group_multiple_people_name")

    def test_call_semantic_node_local_intent_does_not_treat_generic_placeholder_as_person_name(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "group_multiple_people_name",
            "eine person",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 0)
        self.assertEqual(parsed["source"], "local_parser")
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "group_multiple_people_name")

    def test_call_semantic_node_local_intent_falls_back_safely_on_invalid_label(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"visual"}')
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_known_branch",
            "vielleicht bekannt",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 1)
        self.assertEqual(parsed["source"], "local_intent_router_fallback")
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_known_branch")
        self.assertIn("sicherer Fallback", parsed["reason"])

    def test_call_semantic_node_local_intent_falls_back_safely_on_invalid_json(self) -> None:
        fake_router = self._FakeLocalIntentRouter("kein json")
        parsed, decision = call_semantic_node(
            None,
            "",
            "dark_known_branch",
            "vielleicht",
            local_intent_router=fake_router,
        )
        self.assertEqual(fake_router.calls, 1)
        self.assertEqual(parsed["source"], "local_intent_router_fallback")
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_known_branch")
        self.assertIn("could not parse JSON", parsed["error"])

    def test_local_parser_blocks_plain_yes_for_group_person_trigger_reason(self) -> None:
        decision = _local_session_decision("group_person_trigger_reason", "ja")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "group_person_trigger_reason")

    def test_should_use_contextual_same_node_reply_for_group_explanation_nodes(self) -> None:
        decision = validate_semantic_decision(
            "group_person_trigger_role",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "group_person_trigger_role",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        self.assertTrue(_should_use_contextual_same_node_reply("group_person_trigger_role", decision))

    def test_contextual_same_node_reply_uses_model_and_runtime_context_for_text_unclear(self) -> None:
        decision = validate_semantic_decision(
            "group_person_trigger_role",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "group_person_trigger_role",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        fake_client = self._FakeClient(
            "Du hast gerade beschrieben, dass Peter dich auslacht. Spuer jetzt nach: Geht dieses Ausgelachtwerden direkt von Peter aus, oder steht Peter eher fuer die Gruppendynamik?"
        )
        runtime_slots = {
            "named_person": "Peter",
            "group_person_trigger_reason": "ich weiss nicht er lacht mich aus",
        }
        reply = _contextual_same_node_reply(
            fake_client,
            "ft:test",
            "group_person_trigger_role",
            decision,
            customer_message="wie meinst du das?",
            clarify_attempt=0,
            session_context="",
            runtime_slots=runtime_slots,
            silence=False,
        )
        self.assertIn("Peter", reply)
        self.assertIn("auslacht", reply)
        payload = fake_client.chat.completions.last_kwargs["messages"][1]["content"]
        self.assertIn("group_person_trigger_reason", payload)
        self.assertIn("ich weiss nicht er lacht mich aus", payload)

    def test_contextual_same_node_reply_falls_back_without_client(self) -> None:
        decision = validate_semantic_decision(
            "group_person_trigger_role",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "group_person_trigger_role",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        reply = _contextual_same_node_reply(
            None,
            "",
            "group_person_trigger_role",
            decision,
            customer_message="",
            clarify_attempt=0,
            session_context="",
            runtime_slots={"named_person": "Peter"},
            silence=True,
        )
        self.assertIn("Peter", reply)
        self.assertNotIn("Gruppendynamik", reply)
        self.assertNotIn("in dieser Gruppe", reply)

    def test_contextual_same_node_reply_turns_acknowledgement_into_direct_content_request(self) -> None:
        decision = validate_semantic_decision(
            "origin_person_name",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        reply = _contextual_same_node_reply(
            None,
            "",
            "origin_person_name",
            decision,
            customer_message="ja ich erkenne wer es ist",
            clarify_attempt=1,
            session_context="",
            runtime_slots={"trigger_focus_ref": "die person"},
            silence=False,
        )
        self.assertIn("Namen oder eine kurze Beschreibung", reply)
        self.assertNotIn("ersten Eindruck", reply)

    def test_contextual_same_node_reply_sanitizes_meta_step_tail(self) -> None:
        decision = validate_semantic_decision(
            "origin_person_name",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        fake_client = self._FakeClient(
            "Ich meine: Nenne bitte den Namen oder eine kurze Beschreibung der Person. Zurueck bei diesem Schritt."
        )
        reply = _contextual_same_node_reply(
            fake_client,
            "ft:test",
            "origin_person_name",
            decision,
            customer_message="wie meinst du das?",
            clarify_attempt=0,
            session_context="",
            runtime_slots={"trigger_focus_ref": "die person"},
            silence=False,
        )
        self.assertIn("Nenne bitte den Namen", reply)
        self.assertNotIn("Zurueck bei diesem Schritt", reply)

    def test_dynamic_no_reply_progresses_for_phase2_ready(self) -> None:
        decision = _local_session_decision("session_phase2_ready", "nein")
        assert decision is not None
        first = _dynamic_same_node_reply(
            "session_phase2_ready",
            decision,
            0,
            "Gut. Sobald du bereit bist, sag einfach Ja.",
        )
        second = _dynamic_same_node_reply(
            "session_phase2_ready",
            decision,
            1,
            "Gut. Sobald du bereit bist, sag einfach Ja.",
        )
        self.assertNotEqual(first, second)
        self.assertIn("brauchst", second)

    def test_global_uncertainty_maps_to_unclear_in_content_node(self) -> None:
        decision = _local_session_decision("dark_scene_immediate_feeling", "weiss nicht")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_scene_immediate_feeling")

    def test_global_uncertainty_maps_to_unclear_in_known_vs_new_node(self) -> None:
        decision = _local_session_decision("dark_known_branch", "ich bin mir nicht sicher")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        self.assertEqual(decision.next_node, "dark_known_branch")

    def test_dynamic_unclear_reply_escalates_therapeutically(self) -> None:
        decision = SimpleNamespace(intent="unclear")
        first = _dynamic_same_node_reply(
            "dark_known_branch",
            decision,
            0,
            "Nimm kurz wahr, ob dir dieses Gefuehl bereits bekannt vorkommt oder ob du ihm hier zum ersten Mal so begegnest.",
        )
        second = _dynamic_same_node_reply(
            "dark_known_branch",
            decision,
            1,
            "Nimm kurz wahr, ob dir dieses Gefuehl bereits bekannt vorkommt oder ob du ihm hier zum ersten Mal so begegnest.",
        )
        third = _dynamic_same_node_reply(
            "dark_known_branch",
            decision,
            2,
            "Nimm kurz wahr, ob dir dieses Gefuehl bereits bekannt vorkommt oder ob du ihm hier zum ersten Mal so begegnest.",
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertIn("ersten Eindruck", second)
        self.assertIn("ist das in Ordnung", third)

    def test_empty_input_reply_varies_across_first_three_attempts_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]

        offenders: list[str] = []
        for node_id in semantic_node_ids:
            replies = [_empty_input_reply(node_id, attempt) for attempt in range(3)]
            if len(set(replies)) < 3:
                offenders.append(node_id)

        self.assertEqual(offenders, [])

    def test_dynamic_unclear_reply_varies_across_first_three_attempts_for_all_supported_nodes(self) -> None:
        offenders: list[str] = []
        for node_id in available_node_ids():
            spec = get_node_spec(node_id)
            if isinstance(spec, ScriptNodeSpec) or "unclear" not in spec.allowed_intents:
                continue
            decision = SimpleNamespace(intent="unclear")
            fallback = spec.same_node_replies.get("unclear", "")
            replies = [
                _dynamic_same_node_reply(node_id, decision, attempt, fallback, {})
                for attempt in range(3)
            ]
            if len(set(replies)) < 3:
                offenders.append(node_id)

        self.assertEqual(offenders, [])

    def test_dynamic_support_reply_varies_across_first_three_attempts_for_all_supported_nodes(self) -> None:
        offenders: list[str] = []
        for node_id in available_node_ids():
            spec = get_node_spec(node_id)
            if isinstance(spec, ScriptNodeSpec) or "support_needed" not in spec.allowed_intents:
                continue
            decision = SimpleNamespace(intent="support_needed")
            fallback = spec.same_node_replies.get("support_needed", "")
            replies = [
                _dynamic_same_node_reply(node_id, decision, attempt, fallback, {})
                for attempt in range(3)
            ]
            if len(set(replies)) < 3:
                offenders.append(node_id)

        self.assertEqual(offenders, [])

    def test_dark_scene_feeling_support_reply_deintensifies_instead_of_deepening(self) -> None:
        decision = SimpleNamespace(intent="support_needed")
        fallback = script_reply_for_decision("dark_scene_feeling_intensity", decision)
        first = _dynamic_same_node_reply("dark_scene_feeling_intensity", decision, 0, fallback, {})
        second = _dynamic_same_node_reply("dark_scene_feeling_intensity", decision, 1, fallback, {})
        third = _dynamic_same_node_reply("dark_scene_feeling_intensity", decision, 2, fallback, {})

        self.assertIn("nicht tiefer", first)
        self.assertIn("Atem", first)
        self.assertNotIn("tiefer in die Szene hinein", first)
        self.assertIn("Stuhl", second)
        self.assertIn("nicht verstaerken", third)

    def test_dark_scene_immediate_feeling_support_reply_stabilizes_first(self) -> None:
        decision = SimpleNamespace(intent="support_needed")
        fallback = script_reply_for_decision("dark_scene_immediate_feeling", decision)

        self.assertIn("Abstand", fallback)
        self.assertIn("Atem", fallback)
        self.assertNotIn("Spuer noch einmal ganz unmittelbar in diesen Moment hinein", fallback)

    def test_dark_scene_immediate_feeling_unclear_reply_no_longer_lists_example_feelings(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_immediate_feeling",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "dark_scene_immediate_feeling",
                "confidence": 1.0,
                "reason": "Unklar.",
            },
        )
        reply = script_reply_for_decision("dark_scene_immediate_feeling", decision)
        self.assertIn("nicht, wie stark es ist", reply)
        self.assertNotIn("Angst, Druck, Enge, Wut, Scham oder Traurigkeit", reply)

    def test_dynamic_no_reply_varies_across_first_three_attempts_for_all_supported_nodes(self) -> None:
        offenders: list[str] = []
        for node_id in available_node_ids():
            spec = get_node_spec(node_id)
            if isinstance(spec, ScriptNodeSpec) or "no" not in spec.allowed_intents:
                continue
            decision = SimpleNamespace(intent="no")
            fallback = spec.same_node_replies.get("no", "")
            replies = [
                _dynamic_same_node_reply(node_id, decision, attempt, fallback, {})
                for attempt in range(3)
            ]
            if len(set(replies)) < 3:
                offenders.append(node_id)

        self.assertEqual(offenders, [])

    def test_global_uncertainty_matrix_stays_on_safe_intents_for_all_semantic_nodes(self) -> None:
        samples = ("weiss nicht", "bin mir nicht sicher", "keine ahnung")
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]

        for sample in samples:
            for node_id in semantic_node_ids:
                with self.subTest(sample=sample, node_id=node_id):
                    decision = _local_session_decision(node_id, sample)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertIn(decision.intent, {"unclear", "support_needed"})

    def test_scene_found_question_hint_is_not_resource_based(self) -> None:
        hint = QUESTION_ANSWER_HINTS["session_phase2_scene_found"]
        self.assertIn("nicht um eine Ressource", hint)
        self.assertIn("Verlangen nach der Zigarette", hint)

    def test_phase2_scene_guidance_mentions_cigarette_craving(self) -> None:
        self.assertIn("Verlangen nach einer Zigarette", PHASE2_SCENE_GUIDANCE_SCRIPT)
        self.assertIn("Sucht", PHASE2_SCENE_GUIDANCE_SCRIPT)

    def test_extract_scale_value_supports_digits_and_words(self) -> None:
        self.assertEqual(_extract_scale_value("9"), "9")
        self.assertEqual(_extract_scale_value("bei acht"), "8")
        self.assertEqual(_extract_scale_value("ich glaube fuenf"), "5")

    def test_scale_confirmation_prefix_for_post_scale_scripts(self) -> None:
        self.assertEqual(_scale_confirmation_prefix("session_phase2_post_scale_before_script", "9"), "Okay, 9.")
        self.assertEqual(_scale_confirmation_prefix("session_phase2_post_scale_after_script", "5"), "Gut, jetzt bei 5.")

    def test_extract_named_person_label_normalizes_simple_name(self) -> None:
        self.assertEqual(_extract_named_person_label("paul"), "Paul")
        self.assertEqual(_extract_named_person_label("das ist paul"), "Paul")

    def test_extract_named_person_label_rejects_group_phrase(self) -> None:
        self.assertIsNone(_extract_named_person_label("die menschen wo rauchen"))
        self.assertIsNone(_extract_named_person_label("die freunde auf dem pausenhof"))

    def test_extract_named_person_label_rejects_nonhuman_trigger(self) -> None:
        self.assertIsNone(_extract_named_person_label("der rauch"))
        self.assertIsNone(_extract_named_person_label("die situation auf dem schulhof"))

    def test_extract_person_identity_label_reads_embedded_role_phrase(self) -> None:
        self.assertEqual(sandbox._extract_person_identity_label("ich sehe die person es ist ein freund"), "Ein Freund")

    def test_extract_person_identity_label_trims_narrative_tail(self) -> None:
        self.assertEqual(
            sandbox._extract_person_identity_label("ich sehe mein freund vor mir stehen er raucht und lacht"),
            "Mein Freund",
        )

    def test_extract_person_identity_label_rejects_pronoun_reference(self) -> None:
        self.assertIsNone(sandbox._extract_person_identity_label("vermutlich er"))
        self.assertIsNone(sandbox._extract_person_identity_label("vermtulich beim ihm"))

    def test_extract_named_person_label_rejects_generic_person_status_replies(self) -> None:
        for phrase in GENERIC_PERSON_STATUS_REPLIES:
            with self.subTest(phrase=phrase):
                self.assertIsNone(_extract_named_person_label(phrase))

    def test_classify_focus_reference_distinguishes_person_group_and_other(self) -> None:
        self.assertEqual(_classify_focus_reference("peter"), "person")
        self.assertEqual(_classify_focus_reference("mein vater"), "person")
        self.assertEqual(_classify_focus_reference("die menschen wo rauchen"), "group")
        self.assertEqual(_classify_focus_reference("die clique auf dem pausenhof"), "group")
        self.assertEqual(_classify_focus_reference("der rauch"), "other")
        self.assertEqual(_classify_focus_reference("die situation auf dem schulhof"), "other")
        self.assertEqual(_classify_focus_reference("weiss es nicht"), "unknown")

    def test_classify_focus_reference_matrix(self) -> None:
        cases = {
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
                "mein bruder",
                "meine oma",
                "frau meier",
                "der typ mit der zigarette",
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
                "die anderen da hinten",
                "ein haufen leute",
                "die aus der raucherecke",
                "alle am fenster",
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
                "der gestank",
                "das rot",
                "die spannung im raum",
                "rauchgeruch",
                "ein komisches gefuehl",
                "die farbe blau",
            ],
        }
        for expected_kind, phrases in cases.items():
            for phrase in phrases:
                with self.subTest(expected_kind=expected_kind, phrase=phrase):
                    self.assertEqual(_classify_focus_reference(phrase), expected_kind)

    def test_extract_named_person_label_matrix(self) -> None:
        positive_cases = {
            "peter": "Peter",
            "paul": "Paul",
            "das ist paul": "Paul",
            "mein vater": "Mein Vater",
            "mein bruder": "Mein Bruder",
            "meine oma": "Meine Oma",
            "ein freund": "Ein Freund",
            "eine freundin": "Eine Freundin",
            "die lehrerin": "Lehrerin",
            "der chef": "Chef",
            "frau meier": "Frau Meier",
            "er heisst peter": "Peter",
        }
        for phrase, expected in positive_cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(_extract_named_person_label(phrase), expected)

        negative_cases = [
            "die menschen wo rauchen",
            "die clique auf dem pausenhof",
            "meine freunde",
            "der rauch",
            "der geruch",
            "die situation auf dem schulhof",
            "das lachen",
            "die raucherecke",
            "die anderen da hinten",
            "rauchgeruch",
            "die farbe blau",
            "vermutlich die person dort",
            "die person dort",
            "eine frau",
            "alles klar",
            "ich verstehe",
            "etwas anderes",
        ]
        for phrase in [*negative_cases, *GENERIC_PERSON_STATUS_REPLIES]:
            with self.subTest(phrase=phrase):
                self.assertIsNone(_extract_named_person_label(phrase))

    def test_extract_named_person_label_rejects_non_identity_single_word_nouns(self) -> None:
        for phrase in ("spaghetti", "pizza", "wetter", "druck"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(_extract_named_person_label(phrase))

    def test_classify_focus_reference_rejects_acknowledgement_only_replies(self) -> None:
        for phrase in ("alles klar", "ich verstehe", "ja ich erkenne wer es ist"):
            with self.subTest(phrase=phrase):
                self.assertEqual(_classify_focus_reference(phrase), "unknown")

    def test_classify_focus_reference_distinguishes_generic_visual_placeholder_from_unspecific_person(self) -> None:
        self.assertEqual(_classify_focus_reference("ich sehe etwas"), "unknown")
        self.assertEqual(_classify_focus_reference("ich sehe jemanden"), "person")

    def test_display_trigger_focus_ref_v2_edge_cases(self) -> None:
        cases = {
            "die anderen da hinten": "diese Gruppe",
            "ein haufen leute": "diese Gruppe",
            "die aus der raucherecke": "diese Gruppe",
            "der gestank": "der gestank",
            "die farbe blau": "die farbe blau",
            "eine person": "diese Person",
            "die person": "diese Person",
            "vermutlich die person dort": "diese Person",
            "die person vermutlich": "diese Person",
            "wahrscheinlich fritz": "diese Person",
            "mein vater vermutlich": "diese Person",
            "die person dort": "diese Person",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(_render_runtime_text("{trigger_focus_ref}", {"trigger_focus_ref": phrase}), expected)

    def test_capture_runtime_slots_stores_named_person(self) -> None:
        decision = validate_semantic_decision(
            "group_representative_name",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "group_bring_person_forward",
                "confidence": 1.0,
                "reason": "Name genannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("group_representative_name", "paul", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Paul")

    def test_capture_runtime_slots_activates_group_loop_for_multiple_people(self) -> None:
        decision = validate_semantic_decision(
            "group_multiple_people_name",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "group_bring_person_forward",
                "confidence": 1.0,
                "reason": "Name genannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("group_multiple_people_name", "anna", decision, runtime_slots)
        self.assertEqual(runtime_slots["group_loop_active"], "true")
        self.assertEqual(runtime_slots["named_person"], "Anna")

    def test_route_runtime_next_node_sends_multiple_people_back_into_loop(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_self_understands",
            {
                "intent": "yes",
                "action": "transition",
                "next_node": "group_resolution_complete",
                "confidence": 1.0,
                "reason": "Hilft weiter.",
            },
        )
        routed = _route_runtime_next_node(
            "person_switch_self_understands",
            decision,
            {"group_loop_active": "true"},
        )
        self.assertEqual(routed, "group_next_person_check")

    def test_capture_runtime_slots_deactivates_group_loop_when_no_more_people(self) -> None:
        decision = validate_semantic_decision(
            "group_next_person_check",
            {
                "intent": "no",
                "action": "transition",
                "next_node": "group_resolution_complete",
                "confidence": 1.0,
                "reason": "Keine weitere Person offen.",
            },
        )
        runtime_slots = {"group_loop_active": "true"}
        _capture_runtime_slots("group_next_person_check", "nein", decision, runtime_slots)
        self.assertEqual(runtime_slots["group_loop_active"], "false")

    def test_render_runtime_text_inserts_named_person(self) -> None:
        rendered = _render_runtime_text(
            "Gut. Dann hol jetzt {named_person} direkt zu dir.",
            {"named_person": "Paul"},
        )
        self.assertEqual(rendered, "Gut. Dann hol jetzt Paul direkt zu dir.")

    def test_render_runtime_text_humanizes_messy_group_trigger_focus_reference(self) -> None:
        rendered = _render_runtime_text(
            "Gut. Dann holen wir jetzt genau diese Gruppe naeher heran: {trigger_focus_ref}.",
            {"trigger_focus_ref": "ich glaube die freunde die dort am rauchen sind und mich auslachen"},
        )
        self.assertEqual(rendered, "Gut. Dann holen wir jetzt genau diese Gruppe naeher heran: diese Gruppe.")

    def test_render_runtime_text_humanizes_short_group_trigger_focus_reference(self) -> None:
        rendered = _render_runtime_text(
            "Gut. Dann holen wir jetzt genau diese Gruppe naeher heran: {trigger_focus_ref}.",
            {"trigger_focus_ref": "die menschen wo rauchen"},
        )
        self.assertEqual(rendered, "Gut. Dann holen wir jetzt genau diese Gruppe naeher heran: diese Gruppe.")

    def test_render_runtime_text_fills_cigarettes_per_day_from_profile_slot(self) -> None:
        rendered = _render_runtime_text(
            "{anzahl_zigaretten_pro_tag} Mal pro Tag",
            {"zigaretten_pro_tag": "17"},
        )
        self.assertEqual(rendered, "17 Mal pro Tag")

    def test_render_runtime_text_falls_back_for_missing_cigarettes_per_day_placeholder(self) -> None:
        rendered = _render_runtime_text(
            "{anzahl_zigaretten_pro_tag} Mal pro Tag",
            {},
        )
        self.assertEqual(rendered, "20 Mal pro Tag")

    def test_render_runtime_text_builds_origin_scene_reflection_from_runtime_slots(self) -> None:
        rendered = _render_runtime_text(
            "{origin_scene_reflection}",
            {
                "dark_scene_age": "12",
                "dark_scene_visual_detail": "ich sehe eine Gruppe Kinder auf dem Pausenhof",
                "dark_scene_immediate_feeling": "Druck in der Brust",
            },
        )
        self.assertIn("12 Jahre alt", rendered)
        self.assertIn("eine Gruppe Kinder auf dem Pausenhof", rendered)
        self.assertIn("Druck in der Brust", rendered)

    def test_render_runtime_text_uses_origin_scene_reflection_inside_origin_intro(self) -> None:
        raw_text, _ = render_script_node("dark_origin_terminal")
        rendered = _render_runtime_text(
            raw_text,
            {
                "dark_scene_age": "9",
                "dark_scene_who": "mein Vater",
                "dark_scene_immediate_feeling": "Enge",
            },
        )
        self.assertIn("wichtiger Ursprung", rendered)
        self.assertIn("9 Jahre alt", rendered)
        self.assertIn("mein Vater", rendered)
        self.assertIn("Enge", rendered)

    def test_render_runtime_text_origin_scene_reflection_prefers_refined_person_over_generic_placeholder(self) -> None:
        rendered = _render_runtime_text(
            "{origin_scene_reflection}",
            {
                "dark_scene_age": "15",
                "dark_scene_visual_detail": "eine person",
                "dark_scene_people_who": "ich sehe fritz",
                "dark_scene_immediate_feeling": "ein druck",
            },
        )
        self.assertIn("15 Jahre alt", rendered)
        self.assertIn("fritz", rendered.lower())
        self.assertNotIn("du dort wahrnimmst: eine person", rendered.lower())

    def test_restore_german_umlauts_for_tts_fixes_common_hypnosis_words(self) -> None:
        restored = _restore_german_umlauts_for_tts(
            "Wie nimmst du dieses Gefuehl fuer dich wahr und was spuerst du im Koerper?"
        )
        self.assertIn("Gefühl", restored)
        self.assertIn("für", restored)
        self.assertIn("spürst", restored)
        self.assertIn("Körper", restored)

    def test_restore_german_umlauts_for_tts_does_not_break_plain_ue_words(self) -> None:
        restored = _restore_german_umlauts_for_tts(
            "Neue Szenen fuehlen sich anders an als neue Bilder."
        )
        self.assertEqual(restored, "Neue Szenen fühlen sich anders an als neue Bilder.")

    def test_prepare_tts_text_normalizes_only_for_elevenlabs(self) -> None:
        source = "Wie fuehlt sich das fuer dich an?"
        self.assertEqual(_prepare_tts_text(source, provider="google"), source)
        self.assertEqual(
            _prepare_tts_text(source, provider="elevenlabs"),
            "Wie fühlt sich das für dich an?",
        )

    def test_tts_provider_accepts_elevenlabs_alias(self) -> None:
        with patch.dict("os.environ", {"SESSION_SANDBOX_TTS_PROVIDER": "eleven"}, clear=False):
            self.assertEqual(_tts_provider(), "elevenlabs")

    def test_tts_lead_in_ms_defaults_and_clamps(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_tts_lead_in_ms(), 180)
        with patch.dict("os.environ", {"SESSION_SANDBOX_TTS_LEAD_IN_MS": "700"}, clear=False):
            self.assertEqual(_tts_lead_in_ms(), 500)

    def test_tts_post_block_pause_ms_defaults_and_clamps(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_tts_post_block_pause_ms(), 320)
        with patch.dict("os.environ", {"SESSION_SANDBOX_TTS_POST_BLOCK_PAUSE_MS": "4000"}, clear=False):
            self.assertEqual(_tts_post_block_pause_ms(), 1500)

    def test_prepend_silence_to_pcm_adds_expected_prefix_length(self) -> None:
        pcm = b"\x01\x02" * 10
        prefixed = _prepend_silence_to_pcm(
            pcm,
            sample_rate=22050,
            lead_in_ms=100,
            sample_width=2,
            channels=1,
        )
        expected_prefix_bytes = 22050 * 2 // 10
        self.assertEqual(len(prefixed), len(pcm) + expected_prefix_bytes)
        self.assertEqual(prefixed[:expected_prefix_bytes], b"\x00" * expected_prefix_bytes)
        self.assertEqual(prefixed[expected_prefix_bytes:], pcm)

    def test_print_paged_block_waits_after_spoken_block_when_configured(self) -> None:
        with patch.object(sandbox, "_speak_text", return_value=None) as speak_mock, patch(
            "run_session_sandbox.time.sleep"
        ) as sleep_mock:
            _print_paged_block("SCRIPT", "kurzer text", speak=True, post_block_pause_ms=380)
        speak_mock.assert_called_once_with("kurzer text")
        sleep_mock.assert_called_once_with(0.38)

    def test_render_origin_person_branch_intro_avoids_uninflected_raw_phrase(self) -> None:
        raw_text, _ = render_script_node("origin_person_branch_intro")
        rendered = _render_runtime_text(
            raw_text,
            {"trigger_focus_ref": "mein vater"},
        )
        self.assertNotIn("loest dieses Gefuehl", rendered)
        self.assertIn("in den Fokus", rendered)
        self.assertIn("was zwischen euch in diesem Moment wirkt", rendered)

    def test_render_runtime_text_normalizes_indefinite_named_person_in_script_text(self) -> None:
        rendered = _render_runtime_text(
            "Dann lass {named_person} jetzt naeher zu dir kommen.",
            {"named_person": "Ein Freund"},
        )
        self.assertEqual(rendered, "Dann lass diese Person jetzt naeher zu dir kommen.")

    def test_origin_person_name_question_is_available(self) -> None:
        rendered = _render_runtime_question(
            "origin_person_name",
            {"trigger_focus_ref": "die person"},
        )
        self.assertEqual(rendered, "Wenn du bei dieser Person bleibst: Welche Person ist es genau?")

    def test_origin_person_name_unknown_identity_transitions_neutrally(self) -> None:
        decision = _local_session_decision("origin_person_name", "ich weiss nicht wer das ist", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "unknown_person")
        self.assertEqual(decision.next_node, "origin_person_unknown_intro")

    def test_origin_person_name_accepts_singular_role_description(self) -> None:
        decision = _local_session_decision("origin_person_name", "ein freund", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "origin_person_branch_intro")

    def test_origin_person_name_acknowledgement_does_not_fake_ready(self) -> None:
        for phrase in ("alles klar", "ich verstehe", "ja ich erkenne wer es ist", "ok", *GENERIC_PERSON_STATUS_REPLIES):
            with self.subTest(phrase=phrase):
                decision = _local_session_decision("origin_person_name", phrase, restrict_scope=False)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, "origin_person_name")

    def test_all_named_person_input_nodes_reject_generic_person_status_replies(self) -> None:
        for node_id in NAMED_PERSON_INPUT_NODES:
            for phrase in GENERIC_PERSON_STATUS_REPLIES:
                with self.subTest(node_id=node_id, phrase=phrase):
                    decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "unclear")
                    self.assertEqual(decision.next_node, node_id)

    def test_visual_readiness_nodes_accept_person_visibility_status_replies(self) -> None:
        expected_next = {
            "group_person_ready": "group_person_handoff",
            "person_switch_sees_customer": "person_switch_sees_impact",
        }
        for node_id, next_node in expected_next.items():
            for phrase in PERSON_VISIBILITY_READY_REPLIES:
                with self.subTest(node_id=node_id, phrase=phrase):
                    decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "yes")
                    self.assertEqual(decision.next_node, next_node)

    def test_audio_and_impact_status_replies_route_yes_on_readiness_nodes(self) -> None:
        cases = {
            "person_switch_hears": (AUDIO_CONTACT_READY_REPLIES, "person_switch_sees_customer"),
            "person_switch_sees_impact": (IMPACT_VISIBILITY_READY_REPLIES, "person_switch_why"),
            "person_switch_heard_customer": (HEARD_CUSTOMER_READY_REPLIES, "person_switch_why"),
            "person_switch_self_heard": (HEARD_CUSTOMER_READY_REPLIES, "person_switch_self_understands"),
        }
        for node_id, (phrases, next_node) in cases.items():
            for phrase in phrases:
                with self.subTest(node_id=node_id, phrase=phrase):
                    decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertEqual(decision.intent, "yes")
                    self.assertEqual(decision.next_node, next_node)

    def test_person_switch_negative_status_replies_stay_in_same_node(self) -> None:
        cases = (
            ("person_switch_sees_impact", "ich sehe es nicht"),
            ("person_switch_heard_customer", "ich habe es nicht gehoert"),
        )
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "no")
                self.assertEqual(decision.next_node, node_id)

    def test_status_ready_rules_do_not_expand_to_unrelated_content(self) -> None:
        cases = (
            ("group_person_ready", "der rauch"),
            ("group_person_ready", "ein freund"),
            ("person_switch_sees_customer", "die zigarette"),
            ("person_switch_sees_impact", "ich hoere ihn"),
            ("person_switch_heard_customer", "ich sehe ihn"),
            ("person_switch_hears", "ich sehe ihn"),
        )
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                self.assertIsNone(_local_session_decision(node_id, phrase, restrict_scope=False))

    def test_group_person_ready_status_reply_does_not_overwrite_named_person(self) -> None:
        runtime_slots = {"named_person": "Fritz"}
        decision = _local_session_decision("group_person_ready", "die person steht vor mir", restrict_scope=False)
        self.assertIsNotNone(decision)
        assert decision is not None
        _capture_runtime_slots("group_person_ready", "die person steht vor mir", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Fritz")

    def test_origin_person_name_meta_recognition_does_not_write_fake_named_person(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = _local_session_decision("origin_person_name", "ja ich erkenne die person", restrict_scope=False)
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "unclear")
        _capture_runtime_slots("origin_person_name", "ja ich erkenne die person", decision, runtime_slots)
        self.assertNotIn("named_person", runtime_slots)

    def test_acknowledgement_reply_for_dark_scene_people_who_requests_identity_not_yes_no(self) -> None:
        reply = sandbox._acknowledgement_only_same_node_reply(
            "dark_scene_people_who",
            "ja",
            {"dark_scene_visual_detail": "eine person"},
        )
        self.assertIn("Namen oder eine kurze Beschreibung", reply)
        self.assertIn("noch nicht klar", reply)
        self.assertNotIn("diese Rueckmeldung", reply)

    def test_acknowledgement_reply_for_group_representative_name_requests_direct_person_label(self) -> None:
        reply = sandbox._acknowledgement_only_same_node_reply(
            "group_representative_name",
            "ja",
            {},
        )
        self.assertIn("Namen oder eine kurze Beschreibung", reply)
        self.assertNotIn("diese Rueckmeldung", reply)

    def test_run_interactive_origin_person_name_meta_recognition_does_not_corrupt_person_name(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ja ich erkenne die person", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "origin_person_name",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("intent=unclear action=clarify next=origin_person_name", output)
        self.assertIn("Welche Person ist es genau?", output)
        self.assertNotIn("Dann lass Ja Ich Erkenne Die Person", output)
        self.assertNotIn("Steht Ja Ich Erkenne Die Person", output)

    def test_run_interactive_group_person_ready_accepts_visual_status_reply(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["die person steht vor mir", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "group_person_ready",
                semantic_provider="local-intent",
                initial_runtime_slots={"named_person": "Fritz"},
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("intent=yes action=transition next=group_person_handoff", output)
        self.assertNotIn("intent=unclear", output)

    def test_run_interactive_person_switch_heard_customer_accepts_audio_status_reply(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ich habe es gehoert", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "person_switch_heard_customer",
                semantic_provider="local-intent",
                initial_runtime_slots={"named_person": "Fritz", "customer_name": "Nico"},
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("intent=yes action=transition next=person_switch_why", output)
        self.assertNotIn("intent=unclear", output)

    def test_origin_person_name_extra_silence_reasks_before_warning(self) -> None:
        reply = _empty_input_reply("origin_person_name", 4)
        self.assertIn("Namen oder beschreibe kurz", reply)
        self.assertNotIn("zusammenhaengenden Rahmen fortsetzen", reply)

    def test_origin_person_name_unknown_identity_clears_stale_named_person(self) -> None:
        runtime_slots = {"named_person": "Peter"}
        decision = validate_semantic_decision(
            "origin_person_name",
            {
                "intent": "unknown_person",
                "action": "transition",
                "next_node": "origin_person_unknown_intro",
                "confidence": 1.0,
                "reason": "Noch nicht identifizierbar.",
            },
        )
        _capture_runtime_slots("origin_person_name", "ich weiss nicht wer das ist", decision, runtime_slots)
        self.assertNotIn("named_person", runtime_slots)

    def test_group_person_trigger_reason_updates_named_person_when_name_is_revealed(self) -> None:
        runtime_slots = {"named_person": "diese Person"}
        decision = validate_semantic_decision(
            "group_person_trigger_reason",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "group_person_trigger_role",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("group_person_trigger_reason", "er heisst peter", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Peter")

    def test_group_person_trigger_core_does_not_overwrite_named_person_with_abstract_cause(self) -> None:
        runtime_slots = {"named_person": "Peter"}
        decision = validate_semantic_decision(
            "group_person_trigger_core",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "person_switch_ready_intro",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("group_person_trigger_core", "gruppendruck", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Peter")

    def test_origin_person_unknown_intro_exists(self) -> None:
        text, next_node = render_script_node("origin_person_unknown_intro")
        self.assertIn("wer diese Person genau ist", text)
        self.assertEqual(next_node, "group_person_ready")

    def test_render_runtime_question_contextualizes_group_person_trigger_role(self) -> None:
        rendered = _render_runtime_question(
            "group_person_trigger_role",
            {
                "named_person": "Hansi",
                "group_person_trigger_reason": "er lacht mich aus, weil ich nicht rauche",
            },
        )
        self.assertIn("Du hast gerade beschrieben", rendered)
        self.assertIn("er lacht dich aus, weil du nicht rauchst", rendered)
        self.assertIn("Hansi", rendered)

    def test_render_runtime_question_contextualizes_origin_cause_owner_with_trigger_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "die Gruppe",
            },
        )
        self.assertIn("diese Gruppe", rendered)

    def test_render_runtime_question_contextualizes_origin_cause_owner_with_possessive_person_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "mein freund",
            },
        )
        self.assertIn("dein Freund", rendered)
        self.assertNotIn("Mein Freund", rendered)

    def test_render_runtime_question_contextualizes_origin_cause_owner_with_generic_person_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "eine person",
            },
        )
        self.assertIn("diese Person", rendered)
        self.assertNotIn("eine person", rendered)
        self.assertIn("etwas in dir", rendered)

    def test_render_runtime_question_contextualizes_origin_cause_owner_with_hedged_person_focus_as_generic(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "die person vermutlich",
            },
        )
        self.assertIn("diese Person", rendered)
        self.assertNotIn("Person Vermutlich", rendered)
        self.assertNotIn("die person vermutlich", rendered)

    def test_render_runtime_question_uses_scene_named_person_for_generic_person_trigger_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "die person",
                "scene_named_person": "Fritz",
            },
        )
        self.assertIn("Fritz", rendered)
        self.assertNotIn("diese Person", rendered)

    def test_render_runtime_question_contextualizes_dark_scene_people_who_for_generic_person(self) -> None:
        rendered = _render_runtime_question(
            "dark_scene_people_who",
            {
                "dark_scene_visual_detail": "eine person",
            },
        )
        self.assertEqual(
            rendered,
            "Kannst du schon etwas genauer erkennen, wer diese Person sein koennte? Wenn ja, reicht der Name oder eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
        )

    def test_render_runtime_question_contextualizes_dark_scene_people_who_for_group(self) -> None:
        rendered = _render_runtime_question(
            "dark_scene_people_who",
            {
                "dark_scene_visual_detail": "eine gruppe",
            },
        )
        self.assertEqual(
            rendered,
            "Kannst du schon etwas genauer erkennen, wer diese Personen oder diese Gruppe sein koennten? Wenn ja, reicht eine kurze Beschreibung. Wenn es noch nicht klar ist, sag bitte kurz, dass es noch nicht klar ist.",
        )

    def test_render_runtime_question_contextualizes_origin_trigger_known_branch_with_trigger_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_trigger_known_branch",
            {
                "trigger_focus_ref": "die farbe rot",
            },
        )
        self.assertIn("die farbe rot", rendered)
        self.assertIn("Kennst du dieses Gefuehl schon", rendered)
        self.assertIn("zum ersten Mal", rendered)

    def test_render_runtime_question_contextualizes_origin_scene_relevance_with_trigger_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_scene_relevance",
            {
                "trigger_focus_ref": "die farbe rot",
            },
        )
        self.assertIn("die farbe rot", rendered)
        self.assertIn("hier anschauen oder loesen", rendered)
        self.assertIn("etwas Frueherem", rendered)

    def test_render_runtime_question_contextualizes_origin_self_need_with_trigger_focus(self) -> None:
        rendered = _render_runtime_question(
            "origin_self_need",
            {
                "trigger_focus_ref": "der druck",
            },
        )
        self.assertIn("der druck", rendered)
        self.assertIn("am meisten gefehlt", rendered)

    def test_render_runtime_question_humanizes_messy_group_trigger_focus_reference(self) -> None:
        rendered = _render_runtime_question(
            "origin_cause_owner",
            {
                "trigger_focus_ref": "ich glaube die freunde die dort am rauchen sind und mich auslachen",
            },
        )
        self.assertIn("diese Gruppe", rendered)
        self.assertNotIn("ich glaube", rendered)

    def test_origin_other_target_kind_question_mentions_other_case(self) -> None:
        spec = get_node_spec("origin_other_target_kind")
        assert not isinstance(spec, ScriptNodeSpec)
        self.assertIn("etwas anderes", spec.question_text)

    def test_render_runtime_question_contextualizes_group_person_trigger_core(self) -> None:
        rendered = _render_runtime_question(
            "group_person_trigger_core",
            {
                "named_person": "Hansi",
                "group_person_trigger_reason": "er lacht mich aus, weil ich nicht rauche",
            },
        )
        self.assertIn("Du hast gerade beschrieben", rendered)
        self.assertIn("warum Hansi in dieser Situation so reagiert", rendered)

    def test_render_runtime_question_contextualizes_group_person_trigger_core_with_group_role(self) -> None:
        rendered = _render_runtime_question(
            "group_person_trigger_core",
            {
                "named_person": "Peter",
                "group_person_trigger_reason": "er lacht mich aus",
                "group_person_trigger_role": "von allen",
            },
        )
        self.assertIn("Peter", rendered)
        self.assertIn("fuer diese Gruppe steht", rendered)
        self.assertIn("Wenn das noch nicht klar ist", rendered)

    def test_render_runtime_question_addresses_named_person_after_switch(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_hears",
            {
                "named_person": "Peter",
            },
        )
        self.assertEqual(rendered, "Hallo Peter, hoerst du mich?")

    def test_render_runtime_question_for_group_person_ready_is_simple_state_check(self) -> None:
        rendered = _render_runtime_question(
            "group_person_ready",
            {
                "named_person": "Peter",
            },
        )
        self.assertEqual(rendered, "Steht Peter jetzt klar vor dir?")

    def test_render_runtime_question_for_person_switch_ready_uses_generic_person_reference(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_ready",
            {
                "named_person": "eine person",
            },
        )
        self.assertEqual(rendered, "Bist du bereit, jetzt in die Perspektive von dieser Person zu wechseln?")

    def test_group_person_handoff_routes_into_reason_clarification_first(self) -> None:
        spec = get_node_spec("group_person_handoff")
        self.assertIsInstance(spec, ScriptNodeSpec)
        self.assertEqual(spec.next_node, "group_person_trigger_reason")

    def test_person_switch_ready_intro_explains_step_before_ready_question(self) -> None:
        text, next_node = render_script_node("person_switch_ready_intro")
        rendered = _render_runtime_text(text, {"named_person": "Fritz"})
        self.assertIn("erklaere ich dir kurz den naechsten Schritt", rendered)
        self.assertIn("Perspektive von Fritz", rendered)
        self.assertIn("warum Fritz so reagiert hat", rendered)
        self.assertIn("eigene Perspektive", rendered)
        self.assertEqual(next_node, "person_switch_ready")

    def test_person_switch_ready_intro_uses_grammatical_generic_person_reference(self) -> None:
        text, next_node = render_script_node("person_switch_ready_intro")
        rendered = _render_runtime_text(text, {"named_person": "eine person"})
        self.assertIn("Perspektive von dieser Person", rendered)
        self.assertIn("was bei dieser Person", rendered)
        self.assertEqual(next_node, "person_switch_ready")

    def test_render_runtime_question_for_person_switch_sees_customer_uses_customer_name(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_sees_customer",
            {
                "named_person": "Fritz",
                "customer_name": "Nico",
            },
        )
        self.assertEqual(rendered, "Siehst du, dass Nico vor dir steht?")

    def test_render_runtime_question_for_person_switch_sees_customer_falls_back_to_generic_counterpart(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_sees_customer",
            {
                "named_person": "Fritz",
            },
        )
        self.assertEqual(rendered, "Siehst du, dass dein Gegenueber vor dir steht?")

    def test_render_runtime_question_for_person_switch_impact_and_trigger_use_customer_name(self) -> None:
        impact = _render_runtime_question(
            "person_switch_sees_impact",
            {
                "named_person": "Fritz",
                "customer_name": "Nico",
                "dark_scene_immediate_feeling": "druck",
            },
        )
        aware = _render_runtime_question(
            "person_switch_aware_trigger",
            {
                "named_person": "Fritz",
                "customer_name": "Nico",
            },
        )
        self.assertEqual(
            impact,
            "Siehst du, dass Nico vor dir steht und dass dein Verhalten in diesem Moment bei Nico gerade druck ausloest?",
        )
        self.assertEqual(aware, "Ist dir bewusst, dass genau dieser Moment spaeter fuer Nico zu einem Ausloeser fuer das Rauchen wird?")

    def test_render_runtime_question_for_person_switch_heard_customer_uses_known_feeling_and_customer_name(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_heard_customer",
            {
                "named_person": "Fritz",
                "customer_name": "Nico",
                "dark_scene_immediate_feeling": "druck",
            },
        )
        self.assertIn("Nico", rendered)
        self.assertIn("druck", rendered)
        self.assertIn("Rauchen", rendered)

    def test_render_runtime_question_for_person_switch_why_uses_named_person_and_customer_name(self) -> None:
        rendered = _render_runtime_question(
            "person_switch_why",
            {
                "named_person": "Fritz",
                "customer_name": "Nico",
            },
        )
        self.assertIn("Perspektive von Fritz", rendered)
        self.assertIn("Nico", rendered)
        self.assertIn("wie geht es dir dort", rendered)
        self.assertIn("warum reagierst du Nico gegenueber", rendered)

    def test_render_person_switch_intro_mentions_customer_name_when_available(self) -> None:
        text, next_node = render_script_node("person_switch_intro")
        rendered = _render_runtime_text(text, {"named_person": "Fritz", "customer_name": "Nico"})
        self.assertIn("kannst zugleich Nico vor dir wahrnehmen", rendered)
        self.assertEqual(next_node, "person_switch_hears")

    def test_render_runtime_question_contextualizes_group_person_trigger_role_and_reflects_reason(self) -> None:
        rendered = _render_runtime_question(
            "group_person_trigger_role",
            {
                "named_person": "Fritz",
                "group_person_trigger_reason": "das er mich auslacht",
            },
        )
        self.assertIn("dass er dich auslacht", rendered)
        self.assertNotIn("das er mich", rendered)
        self.assertNotIn("in dieser Gruppe", rendered)

    def test_dark_origin_terminal_now_flows_into_origin_event_then_trigger_source(self) -> None:
        text, next_node = render_script_node("dark_origin_terminal")
        self.assertIn("Gut. Dann bleiben wir jetzt in genau dieser Szene", text)
        self.assertIn("wichtiger Ursprung", text)
        self.assertNotIn("noch weiter zu etwas Frueherem fuehrt", text)
        self.assertEqual(next_node, "dark_scene_happening")

    def test_dynamic_same_node_reply_contextualizes_person_switch_impact_with_known_customer_effect(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_sees_impact",
            {
                "intent": "no",
                "action": "support",
                "next_node": "person_switch_sees_impact",
                "confidence": 1.0,
                "reason": "Noch nicht klar.",
            },
        )
        reply = _dynamic_same_node_reply(
            "person_switch_sees_impact",
            decision,
            0,
            "fallback",
            runtime_slots={"named_person": "Fritz", "customer_name": "Nico", "dark_scene_immediate_feeling": "druck"},
        )
        self.assertIn("Nico", reply)
        self.assertIn("druck", reply)
        self.assertIn("Ja oder Nein", reply)

    def test_dynamic_same_node_reply_contextualizes_person_switch_heard_customer_with_known_customer_effect(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_heard_customer",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "person_switch_heard_customer",
                "confidence": 1.0,
                "reason": "Noch nicht klar.",
            },
        )
        reply = _dynamic_same_node_reply(
            "person_switch_heard_customer",
            decision,
            0,
            "fallback",
            runtime_slots={"named_person": "Fritz", "customer_name": "Nico", "dark_scene_immediate_feeling": "druck"},
        )
        self.assertIn("Nico", reply)
        self.assertIn("druck", reply)
        self.assertIn("Ja oder Nein", reply)

    def test_dynamic_same_node_reply_for_person_switch_why_requests_free_explanation(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_why",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "person_switch_why",
                "confidence": 1.0,
                "reason": "Noch nicht klar.",
            },
        )
        reply = _dynamic_same_node_reply(
            "person_switch_why",
            decision,
            0,
            "fallback",
            runtime_slots={"named_person": "Fritz", "customer_name": "Nico"},
        )
        self.assertIn("Perspektive von Fritz", reply)
        self.assertIn("Nico", reply)
        self.assertIn("in dir", reply)
        self.assertNotIn("in Fritz", reply)
        self.assertNotIn("Ja oder Nein", reply)

    def test_build_initial_runtime_slots_defaults_customer_name_to_nico(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(sandbox._build_initial_runtime_slots(None), {"customer_name": "Nico"})

    def test_origin_self_resolution_intro_now_flows_into_need_question(self) -> None:
        text, next_node = render_script_node("origin_self_resolution_intro")
        self.assertIn("aus dir selbst heraus entstanden", text)
        self.assertEqual(next_node, "origin_self_need")

    def test_origin_self_release_intro_hands_off_into_common_present_self_block(self) -> None:
        text, next_node = render_script_node("origin_self_release_intro")
        self.assertIn("genau das, was dort gefehlt hat", text)
        self.assertEqual(next_node, "phase4_common_present_self_intro")

    def test_capture_runtime_slots_stores_origin_trigger_focus_reference(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_source", "die clique aus der pause", decision, runtime_slots)
        self.assertEqual(runtime_slots["trigger_focus_ref"], "die clique aus der pause")

    def test_capture_runtime_slots_stores_origin_self_need(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "origin_self_need",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_self_release_intro",
                "confidence": 1.0,
                "reason": "Fehlendes Beduerfnis erkannt.",
            },
        )
        _capture_runtime_slots("origin_self_need", "schutz und halt", decision, runtime_slots)
        self.assertEqual(runtime_slots["origin_self_need"], "schutz und halt")

    def test_origin_other_target_kind_person_derives_named_person_from_trigger_focus(self) -> None:
        runtime_slots = {"trigger_focus_ref": "hansi"}
        decision = validate_semantic_decision(
            "origin_other_target_kind",
            {
                "intent": "person",
                "action": "transition",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Person erkannt.",
            },
        )
        _capture_runtime_slots("origin_other_target_kind", "eine bestimmte person", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Hansi")

    def test_capture_runtime_slots_stores_named_person_from_origin_trigger_source(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_source", "mein vater", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Mein Vater")

    def test_capture_runtime_slots_stores_scene_named_person_from_dark_scene_people_who(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_people_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Konkrete Person erkannt.",
            },
        )
        _capture_runtime_slots("dark_scene_people_who", "ich sehe fritz", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Fritz")
        self.assertEqual(runtime_slots["scene_named_person"], "Fritz")
        self.assertEqual(runtime_slots["dark_scene_visual_detail"], "ich sehe fritz")

    def test_capture_runtime_slots_stores_scene_named_person_from_embedded_role_phrase(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_people_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Konkrete Person erkannt.",
            },
        )
        _capture_runtime_slots("dark_scene_people_who", "ich sehe die person es ist ein freund", decision, runtime_slots)
        self.assertEqual(runtime_slots["named_person"], "Ein Freund")
        self.assertEqual(runtime_slots["scene_named_person"], "Ein Freund")
        reflected = _render_runtime_text(
            "{origin_scene_reflection}",
            runtime_slots | {"dark_scene_age": "15", "dark_scene_immediate_feeling": "druck"},
        )
        self.assertIn("Ein Freund", reflected)

    def test_capture_runtime_slots_preserves_scene_named_person_for_hedged_origin_trigger_source(self) -> None:
        runtime_slots: dict[str, str] = {
            "named_person": "Fritz",
            "scene_named_person": "Fritz",
        }
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_source", "ich vermute fritz", decision, runtime_slots)
        self.assertEqual(runtime_slots["trigger_focus_ref"], "Fritz")
        self.assertEqual(runtime_slots["named_person"], "Fritz")
        self.assertEqual(runtime_slots["scene_named_person"], "Fritz")

    def test_capture_runtime_slots_reuses_scene_named_person_for_pronoun_origin_trigger_source(self) -> None:
        runtime_slots: dict[str, str] = {
            "scene_named_person": "Ein Freund",
        }
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_source", "vermutlich er", decision, runtime_slots)
        self.assertEqual(runtime_slots["trigger_focus_ref"], "Ein Freund")
        self.assertEqual(runtime_slots["named_person"], "Ein Freund")
        self.assertEqual(runtime_slots["scene_named_person"], "Ein Freund")

    def test_capture_runtime_slots_preserves_scene_named_person_for_generic_origin_cause_owner_other_party(self) -> None:
        runtime_slots: dict[str, str] = {
            "trigger_focus_ref": "die person",
            "named_person": "Fritz",
            "scene_named_person": "Fritz",
        }
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        _capture_runtime_slots("origin_cause_owner", "die person", decision, runtime_slots)
        self.assertEqual(runtime_slots["trigger_focus_ref"], "Fritz")
        self.assertEqual(runtime_slots["named_person"], "Fritz")
        self.assertEqual(runtime_slots["scene_named_person"], "Fritz")

    def test_canonicalize_person_focus_ref_does_not_reuse_scene_name_for_jemand_anderes(self) -> None:
        self.assertEqual(
            sandbox._canonicalize_person_focus_ref({"scene_named_person": "Fritz"}, "jemand anderes"),
            ("jemand anderes", None),
        )

    def test_capture_runtime_slots_stores_dark_known_state(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_known_branch",
            {
                "intent": "new",
                "action": "transition",
                "next_node": "dark_origin_terminal",
                "confidence": 1.0,
                "reason": "Erstes Auftreten erkannt.",
            },
        )
        _capture_runtime_slots("dark_known_branch", "zum ersten mal", decision, runtime_slots)
        self.assertEqual(runtime_slots["dark_known_state"], "new")

    def test_capture_runtime_slots_stores_origin_trigger_known_state(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "origin_trigger_known_branch",
            {
                "intent": "known",
                "action": "transition",
                "next_node": "dark_backtrace_terminal",
                "confidence": 1.0,
                "reason": "Bereits bekannt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_known_branch", "das kenne ich schon", decision, runtime_slots)
        self.assertEqual(runtime_slots["origin_trigger_known_state"], "known")

    def test_capture_runtime_slots_clears_named_person_for_generic_origin_trigger_source(self) -> None:
        runtime_slots: dict[str, str] = {"named_person": "Peter"}
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        _capture_runtime_slots("origin_trigger_source", "vermutlich die person", decision, runtime_slots)
        self.assertEqual(runtime_slots["trigger_focus_ref"], "vermutlich die person")
        self.assertNotIn("named_person", runtime_slots)

    def test_capture_runtime_slots_clears_named_person_for_generic_origin_cause_owner_other_party(self) -> None:
        runtime_slots: dict[str, str] = {
            "trigger_focus_ref": "vermutlich die person",
            "named_person": "Peter",
        }
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        _capture_runtime_slots("origin_cause_owner", "jemand anderes", decision, runtime_slots)
        self.assertNotIn("named_person", runtime_slots)

    def test_route_runtime_next_node_corrects_person_misclassification_for_group_trigger(self) -> None:
        decision = validate_semantic_decision(
            "origin_other_target_kind",
            {
                "intent": "person",
                "action": "transition",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Defensive Korrektur fuer Gruppenbeschreibung.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_other_target_kind",
            decision,
            {"trigger_focus_ref": "die menschen wo rauchen"},
        )
        self.assertEqual(routed, "group_branch_intro")

    def test_route_runtime_next_node_sends_nonhuman_trigger_to_self_resolution(self) -> None:
        decision = validate_semantic_decision(
            "origin_other_target_kind",
            {
                "intent": "person",
                "action": "transition",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Defensive Korrektur fuer Nicht-Personen.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_other_target_kind",
            decision,
            {"trigger_focus_ref": "der rauch"},
        )
        self.assertEqual(routed, "origin_self_resolution_intro")

    def test_route_runtime_next_node_defensively_corrects_focus_reference_matrix(self) -> None:
        person_misclassified_decision = validate_semantic_decision(
            "origin_other_target_kind",
            {
                "intent": "person",
                "action": "transition",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Defensive Korrektur.",
            },
        )
        group_inputs = [
            "die menschen wo rauchen",
            "die clique auf dem pausenhof",
            "meine freunde",
            "die gruppe vor der schule",
        ]
        for phrase in group_inputs:
            with self.subTest(kind="group", phrase=phrase):
                routed = _route_runtime_next_node(
                    "origin_other_target_kind",
                    person_misclassified_decision,
                    {"trigger_focus_ref": phrase},
                )
                self.assertEqual(routed, "group_branch_intro")

        other_inputs = [
            "der rauch",
            "der geruch",
            "die situation auf dem schulhof",
            "das lachen",
            "das verhalten",
        ]
        for phrase in other_inputs:
            with self.subTest(kind="other", phrase=phrase):
                routed = _route_runtime_next_node(
                    "origin_other_target_kind",
                    person_misclassified_decision,
                    {"trigger_focus_ref": phrase},
                )
                self.assertEqual(routed, "origin_self_resolution_intro")

    def test_capture_runtime_slots_stores_group_person_clarifications(self) -> None:
        runtime_slots: dict[str, str] = {}
        for node_id, value in [
            ("group_person_trigger_reason", "er loest in mir Druck aus"),
            ("group_person_trigger_role", "er steht eher stellvertretend fuer die Gruppe"),
            ("group_person_trigger_core", "der Kern ist seine abwertende Ausstrahlung"),
        ]:
            decision = validate_semantic_decision(
                node_id,
                {
                    "intent": "ready",
                    "action": "transition",
                    "next_node": {
                        "group_person_trigger_reason": "group_person_trigger_role",
                        "group_person_trigger_role": "group_person_trigger_core",
                        "group_person_trigger_core": "person_switch_ready_intro",
                    }[node_id],
                    "confidence": 1.0,
                    "reason": "Freie inhaltliche Rueckmeldung erkannt.",
                },
            )
            _capture_runtime_slots(node_id, value, decision, runtime_slots)

        self.assertEqual(runtime_slots["group_person_trigger_reason"], "er loest in mir Druck aus")
        self.assertEqual(runtime_slots["group_person_trigger_role"], "er steht eher stellvertretend fuer die Gruppe")
        self.assertEqual(runtime_slots["group_person_trigger_core"], "der Kern ist seine abwertende Ausstrahlung")

    def test_capture_runtime_slots_marks_audio_pending_for_dark_scene_both(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "both",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Sowohl sehen als auch hoeren erkannt.",
            },
        )
        _capture_runtime_slots("dark_scene_perception", "ich sehe und hoere etwas", decision, runtime_slots)
        self.assertEqual(runtime_slots["dark_audio_pending"], "true")
        self.assertEqual(runtime_slots["dark_scene_perception"], "ich sehe und hoere etwas")

    def test_capture_runtime_slots_clears_audio_pending_after_audio_detail(self) -> None:
        runtime_slots = {"dark_audio_pending": "true"}
        decision = validate_semantic_decision(
            "dark_scene_audio_detail",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Auditive Details beschrieben.",
            },
        )
        _capture_runtime_slots("dark_scene_audio_detail", "ich hoere lachen", decision, runtime_slots)
        self.assertEqual(runtime_slots["dark_audio_pending"], "false")
        self.assertEqual(runtime_slots["dark_scene_audio_detail"], "ich hoere lachen")

    def test_route_runtime_next_node_runs_audio_detail_after_visual_when_both_were_present(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail beschrieben.",
            },
        )
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            {"dark_audio_pending": "true", "dark_scene_visual_detail": "eine schulklasse"},
        )
        self.assertEqual(routed, "dark_scene_audio_detail")

    def test_route_runtime_next_node_asks_people_who_after_visual_people_detail(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail beschrieben.",
            },
        )
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            {"dark_scene_visual_detail": "eine gruppe kinder auf dem pausenhof"},
        )
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_asks_people_who_after_audio_detail_when_visual_people_exist(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_audio_detail",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Auditive Details beschrieben.",
            },
        )
        routed = _route_runtime_next_node(
            "dark_scene_audio_detail",
            decision,
            {"dark_scene_visual_detail": "mein vater und noch andere leute"},
        )
        self.assertEqual(routed, "dark_scene_people_who")

    def test_dark_scene_perception_routes_person_hint_directly_to_people_who(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "visual",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Visueller Zugang erkannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("dark_scene_perception", "ich sehe eine person", decision, runtime_slots)
        routed = _route_runtime_next_node(
            "dark_scene_perception",
            decision,
            runtime_slots,
            "ich sehe eine person",
        )
        self.assertEqual(routed, "dark_scene_people_who")

    def test_local_semantics_map_dark_scene_perception_audio_access(self) -> None:
        decision = _local_session_decision("dark_scene_perception", "ich hoere lachen", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "audio")
        self.assertEqual(decision.next_node, "dark_scene_audio_detail")

    def test_local_semantics_map_dark_scene_perception_other_sense_access(self) -> None:
        decision = _local_session_decision("dark_scene_perception", "ich rieche rauch", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "other_sense")
        self.assertEqual(decision.next_node, "dark_scene_other_sense")

    def test_local_semantics_map_dark_scene_perception_visual_fragments(self) -> None:
        cases = ["ein gebaeude", "blau", "schulhof"]
        for phrase in cases:
            with self.subTest(phrase=phrase):
                decision = _local_session_decision("dark_scene_perception", phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "visual")
                self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_semantics_map_dark_scene_mode_clarify_visual_access(self) -> None:
        decision = _local_session_decision("dark_scene_mode_clarify", "ich sehe meinen vater", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "visual")
        self.assertEqual(decision.next_node, "dark_scene_who")

    def test_local_semantics_map_dark_scene_mode_clarify_audio_access(self) -> None:
        decision = _local_session_decision("dark_scene_mode_clarify", "ich hoere stimmen", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "audio")
        self.assertEqual(decision.next_node, "dark_scene_audio_detail")

    def test_local_semantics_map_dark_scene_mode_clarify_other_sense_access(self) -> None:
        decision = _local_session_decision("dark_scene_mode_clarify", "ich rieche etwas verbranntes", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "other_sense")
        self.assertEqual(decision.next_node, "dark_scene_other_sense")

    def test_local_semantics_map_dark_scene_who_leaves_simple_visual_detail_for_model(self) -> None:
        decision = _local_session_decision("dark_scene_who", "eine person")
        self.assertIsNone(decision)

    def test_local_semantics_freeform_scene_detail_nodes_do_not_accept_unrelated_content_locally(self) -> None:
        cases = [
            ("dark_scene_who", "morgen ist dienstag"),
            ("dark_scene_audio_detail", "ich mag pizza"),
            ("dark_scene_other_sense", "das wetter ist schoen"),
            ("dark_scene_first_spuerbar", "ich mag pizza"),
            ("dark_scene_people_who", "spaghetti"),
            ("dark_scene_happening", "morgen regnet es"),
        ]
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                self.assertIsNone(_local_session_decision(node_id, phrase))

    def test_local_semantics_dark_scene_people_who_accepts_specific_person_label(self) -> None:
        decision = _local_session_decision("dark_scene_people_who", "fritz")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "ready")
        self.assertEqual(decision.next_node, "dark_scene_age")

    def test_local_semantics_keep_generic_sense_placeholders_in_same_node(self) -> None:
        cases = [
            ("dark_scene_audio_detail", "ich hoere etwas"),
            ("dark_scene_audio_detail", "ein geraeusch"),
            ("dark_scene_other_sense", "ich spuere etwas"),
            ("dark_scene_first_spuerbar", "ich spuere etwas"),
            ("dark_scene_immediate_feeling", "ich spuere etwas"),
        ]
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, node_id)

    def test_local_semantics_body_bridge_nodes_accept_clear_late_channel_switches(self) -> None:
        cases = [
            ("dark_scene_other_sense", "ich sehe was"),
            ("dark_scene_other_sense", "ich hoere was"),
            ("dark_scene_other_sense", "dunkel"),
            ("dark_scene_first_spuerbar", "ich sehe was"),
            ("dark_scene_first_spuerbar", "hell"),
        ]
        for node_id, phrase in cases:
            with self.subTest(node_id=node_id, phrase=phrase):
                decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "ready")

    def test_local_semantics_body_bridge_nodes_accept_visual_fragments_but_not_off_topic_meta(self) -> None:
        ready_cases = [
            ("dark_scene_other_sense", "ein gebaeude"),
            ("dark_scene_other_sense", "blau"),
            ("dark_scene_other_sense", "schulhof"),
            ("dark_scene_first_spuerbar", "ein gebaeude"),
            ("dark_scene_first_spuerbar", "blau"),
        ]
        unclear_cases = [
            ("dark_scene_other_sense", "erste mal"),
            ("dark_scene_other_sense", "ich bin am gleichen ort wie vorher"),
            ("dark_scene_first_spuerbar", "erste mal"),
            ("dark_scene_first_spuerbar", "ich bin am gleichen ort wie vorher"),
        ]
        for node_id, phrase in ready_cases:
            with self.subTest(node_id=node_id, phrase=phrase, mode="ready"):
                decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "ready")
        for node_id, phrase in unclear_cases:
            with self.subTest(node_id=node_id, phrase=phrase, mode="unclear"):
                decision = _local_session_decision(node_id, phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, node_id)

    def test_local_semantics_dark_scene_immediate_feeling_only_accepts_real_feelings_locally(self) -> None:
        ready_cases = ["druck eventuell", "angst", "klo im hals", "herzklopfen", "kalt in der brust"]
        unclear_cases = [
            "dunkel",
            "hell",
            "eine person",
            "ein gebaeude",
            "ich sehe was",
            "ich rieche rauch",
            "erste mal",
            "ich bin am gleichen ort wie vorher",
            "gar nichts",
        ]
        for phrase in ready_cases:
            with self.subTest(phrase=phrase, mode="ready"):
                decision = _local_session_decision("dark_scene_immediate_feeling", phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "ready")
        for phrase in unclear_cases:
            with self.subTest(phrase=phrase, mode="unclear"):
                decision = _local_session_decision("dark_scene_immediate_feeling", phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "unclear")
                self.assertEqual(decision.next_node, "dark_scene_immediate_feeling")

    def test_local_semantics_origin_trigger_source_blocks_generic_placeholder_but_accepts_real_trigger(self) -> None:
        generic_decision = _local_session_decision("origin_trigger_source", "ich sehe etwas", restrict_scope=False)
        self.assertIsNotNone(generic_decision)
        assert generic_decision is not None
        self.assertEqual(generic_decision.intent, "unclear")
        self.assertEqual(generic_decision.next_node, "origin_trigger_source")

        for phrase in ("die person", "der druck", "mein vater"):
            with self.subTest(phrase=phrase):
                decision = _local_session_decision("origin_trigger_source", phrase, restrict_scope=False)
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "ready")
                self.assertEqual(decision.next_node, "origin_trigger_known_branch")

    def test_route_runtime_next_node_maps_dark_scene_who_people_detail_to_people_followup(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail erkannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("dark_scene_who", "eine person", decision, runtime_slots)
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            runtime_slots,
            "eine person",
        )
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_maps_dark_scene_who_unspecific_someone_to_people_followup(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail erkannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("dark_scene_who", "ich sehe jemanden", decision, runtime_slots)
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            runtime_slots,
            "ich sehe jemanden",
        )
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_skips_people_followup_for_identified_person_detail(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            {"dark_scene_visual_detail": "mein vater"},
        )
        self.assertEqual(routed, "dark_scene_age")

    def test_route_runtime_next_node_skips_people_followup_for_singular_role_detail(self) -> None:
        decision = validate_semantic_decision(
            "dark_scene_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Visuelles Detail erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "dark_scene_who",
            decision,
            {"dark_scene_visual_detail": "ein freund"},
        )
        self.assertEqual(routed, "dark_scene_age")

    def test_sanitize_customer_facing_answer_removes_meta_step_tail(self) -> None:
        sanitized = _sanitize_customer_facing_answer(
            "Ich meine: Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung. Zurueck bei diesem Schritt."
        )
        self.assertEqual(
            sanitized,
            "Ich meine: Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung.",
        )

    def test_sanitize_customer_facing_answer_removes_immediate_feeling_instruction_tail(self) -> None:
        sanitized = _sanitize_customer_facing_answer(
            "Ich meine: richte die Aufmerksamkeit auf das, was jetzt koerperlich und unmittelbar spuerbar ist. "
            "Aber beschreibe nur das unmittelbare Gefuehl, nicht Gedanken oder Ursachen."
        )
        self.assertEqual(
            sanitized,
            "Ich meine: richte die Aufmerksamkeit auf das, was jetzt koerperlich und unmittelbar spuerbar ist",
        )

    def test_answer_question_in_context_sanitizes_runtime_meta_tail(self) -> None:
        fake_client = self._FakeClient(
            "Ich meine: Schau innerlich, ob die Person erkennbar ist. Wenn es klarer wird, reicht der Name oder eine kurze Beschreibung. Zurueck bei diesem Schritt."
        )
        reply = _answer_question_in_context(
            fake_client,
            "ft:test",
            "origin_person_name",
            "wie meinst du das?",
            runtime_slots={"trigger_focus_ref": "die person"},
        )
        self.assertNotIn("Zurueck bei diesem Schritt", reply)
        self.assertIn("kurze Beschreibung", reply)

    def test_answer_question_in_context_sanitizes_immediate_feeling_instruction_tail(self) -> None:
        fake_client = self._FakeClient(
            "Ich meine: richte die Aufmerksamkeit auf das, was jetzt koerperlich und unmittelbar spuerbar ist, "
            "z. B. Enge oder Druck im Brustkorb. Aber beschreibe nur das unmittelbare Gefuehl, nicht Gedanken oder Ursachen."
        )
        reply = _answer_question_in_context(
            fake_client,
            "ft:test",
            "dark_scene_immediate_feeling",
            "wie meinst du das?",
        )
        self.assertIn("koerperlich und unmittelbar spuerbar", reply)
        self.assertNotIn("nicht Gedanken oder Ursachen", reply)

    def test_route_runtime_next_node_respects_explicit_person_choice_for_origin_other_target_kind(self) -> None:
        decision = validate_semantic_decision(
            "origin_other_target_kind",
            {
                "intent": "person",
                "action": "transition",
                "next_node": "origin_person_name",
                "confidence": 1.0,
                "reason": "Explizite Personeneinordnung erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_other_target_kind",
            decision,
            {"trigger_focus_ref": "die freunde auf dem pausenhof"},
            "eine person",
        )
        self.assertEqual(routed, "origin_person_name")

    def test_local_semantics_map_explicit_other_choice_for_origin_other_target_kind(self) -> None:
        decision = _local_session_decision("origin_other_target_kind", "etwas anderes", restrict_scope=False)
        assert decision is not None
        self.assertEqual(decision.intent, "other")
        self.assertEqual(decision.next_node, "origin_self_resolution_intro")

    def test_route_runtime_next_node_skips_origin_other_target_kind_for_named_person_trigger(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "mein vater", "named_person": "Mein Vater"},
            "die person selber",
        )
        self.assertEqual(routed, "origin_person_branch_intro")

    def test_route_runtime_next_node_skips_repeated_known_check_after_new_origin_decision(self) -> None:
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Ausloeser erkannt.",
            },
        )
        runtime_slots = {"dark_known_state": "new"}
        _capture_runtime_slots("origin_trigger_source", "fritz", decision, runtime_slots)
        routed = _route_runtime_next_node(
            "origin_trigger_source",
            decision,
            runtime_slots,
            "fritz",
        )
        self.assertEqual(routed, "origin_cause_owner")

    def test_route_runtime_next_node_keeps_known_check_without_prior_new_origin_decision(self) -> None:
        decision = validate_semantic_decision(
            "origin_trigger_source",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "origin_trigger_known_branch",
                "confidence": 1.0,
                "reason": "Ausloeser erkannt.",
            },
        )
        runtime_slots: dict[str, str] = {}
        _capture_runtime_slots("origin_trigger_source", "fritz", decision, runtime_slots)
        routed = _route_runtime_next_node(
            "origin_trigger_source",
            decision,
            runtime_slots,
            "fritz",
        )
        self.assertEqual(routed, "origin_trigger_known_branch")

    def test_route_runtime_next_node_skips_origin_other_target_kind_for_generic_person_trigger(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "die person"},
            "die person selber",
        )
        self.assertEqual(routed, "origin_person_name")

    def test_route_runtime_next_node_keeps_generic_person_placeholder_out_of_branch_intro(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "vermutlich die person dort"},
            "die person",
        )
        self.assertEqual(routed, "origin_person_name")

    def test_route_runtime_next_node_keeps_hedged_generic_person_out_of_branch_intro(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "die person vermutlich"},
            "es ist die person",
        )
        self.assertEqual(routed, "origin_person_name")

    def test_route_runtime_next_node_keeps_hedged_specific_person_out_of_branch_intro(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "wahrscheinlich fritz"},
            "es ist die person",
        )
        self.assertEqual(routed, "origin_person_name")

    def test_route_runtime_next_node_skips_origin_other_target_kind_for_group_trigger(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "die gruppe"},
            "die gruppe selber",
        )
        self.assertEqual(routed, "group_branch_intro")

    def test_route_runtime_next_node_keeps_origin_other_target_kind_for_nonhuman_trigger(self) -> None:
        decision = validate_semantic_decision(
            "origin_cause_owner",
            {
                "intent": "someone_else",
                "action": "transition",
                "next_node": "origin_other_target_kind",
                "confidence": 1.0,
                "reason": "Anderer Ausloeser erkannt.",
            },
        )
        routed = _route_runtime_next_node(
            "origin_cause_owner",
            decision,
            {"trigger_focus_ref": "der druck"},
            "jemand anderes",
        )
        self.assertEqual(routed, "origin_other_target_kind")

    def test_route_runtime_next_node_skips_visual_detail_question_when_followup_already_contains_specific_people_scene(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "scene_access_followup",
            {
                "intent": "visual_hell",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Konkretes visuelles Material erkannt.",
            },
        )
        _capture_runtime_slots(
            "scene_access_followup",
            "ich sehe eine gruppe kinder auf dem pausenhof",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("scene_access_followup", decision, runtime_slots)
        self.assertEqual(routed, "dark_scene_people_who")

    def test_route_runtime_next_node_skips_visual_detail_question_when_followup_already_contains_specific_visual_fragment(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "scene_access_followup",
            {
                "intent": "visual_hell",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Konkretes visuelles Fragment erkannt.",
            },
        )
        _capture_runtime_slots(
            "scene_access_followup",
            "ein gebaeude",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("scene_access_followup", decision, runtime_slots)
        self.assertEqual(routed, "dark_scene_age")

    def test_route_runtime_next_node_skips_visual_detail_question_when_perception_already_contains_specific_scene(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "visual",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Konkretes Bild erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_perception",
            "ich sehe meinen vater in der kueche",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_perception", decision, runtime_slots)
        self.assertEqual(routed, "dark_scene_age")

    def test_route_runtime_next_node_keeps_visual_detail_question_when_perception_only_contains_visual_fragment(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "visual",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Visuelles Fragment erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_perception",
            "blau",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_perception", decision, runtime_slots)
        self.assertEqual(routed, "dark_scene_who")

    def test_route_runtime_next_node_skips_audio_detail_question_when_perception_already_contains_specific_audio(self) -> None:
        runtime_slots: dict[str, str] = {}
        decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "audio",
                "action": "transition",
                "next_node": "dark_scene_audio_detail",
                "confidence": 1.0,
                "reason": "Konkrete auditive Wahrnehmung erkannt.",
            },
        )
        _capture_runtime_slots(
            "dark_scene_perception",
            "ich hoere wie sie mich auslachen",
            decision,
            runtime_slots,
        )
        routed = _route_runtime_next_node("dark_scene_perception", decision, runtime_slots)
        self.assertEqual(routed, "dark_scene_age")

    def test_group_person_clarification_happy_path_reaches_person_switch(self) -> None:
        reason_decision = validate_semantic_decision(
            "group_person_trigger_reason",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "group_person_trigger_role",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        role_decision = validate_semantic_decision(
            "group_person_trigger_role",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "group_person_trigger_core",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        core_decision = validate_semantic_decision(
            "group_person_trigger_core",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "person_switch_ready_intro",
                "confidence": 1.0,
                "reason": "Inhalt erkannt.",
            },
        )
        self.assertEqual(reason_decision.next_node, "group_person_trigger_role")
        self.assertEqual(role_decision.next_node, "group_person_trigger_core")
        self.assertEqual(core_decision.next_node, "person_switch_ready_intro")

    def test_person_switch_ready_routes_into_countdown(self) -> None:
        decision = validate_semantic_decision(
            "person_switch_ready",
            {
                "intent": "yes",
                "action": "transition",
                "next_node": "person_switch_intro",
                "confidence": 1.0,
                "reason": "Bereit fuer den Perspektivwechsel.",
            },
        )
        self.assertEqual(decision.next_node, "person_switch_intro")

    def test_local_parser_maps_plain_yes_for_person_switch_ready(self) -> None:
        decision = _local_session_decision("person_switch_ready", "ja")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.intent, "yes")
        self.assertEqual(decision.next_node, "person_switch_intro")

    def test_reengagement_signal_detects_simple_return_messages(self) -> None:
        self.assertTrue(_is_reengagement_signal("ich bin da"))
        self.assertTrue(_is_reengagement_signal("bin wieder da"))
        self.assertTrue(_is_reengagement_signal("ich hoere dich"))
        self.assertFalse(_is_reengagement_signal("menschen"))

    def test_first_cigarette_script_is_split_before_follow_up_question(self) -> None:
        script_text, next_node = render_script_node("phase4_common_first_cigarette")
        follow_up_text, follow_up_next = render_script_node("phase4_common_first_cigarette_consequences")
        self.assertEqual(next_node, "phase4_common_first_cigarette_consequences")
        self.assertEqual(follow_up_next, "phase4_common_feel_after_learning")
        self.assertTrue(script_text)
        self.assertEqual(follow_up_text, PHASE4_COMMON_FIRST_CIGARETTE_CONSEQUENCES)

    def test_run_interactive_continues_after_split_first_cigarette_script_with_speech(self) -> None:
        ready_decision = validate_semantic_decision(
            "phase4_common_feel_after_learning",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "phase4_common_first_drag",
                "confidence": 1.0,
                "reason": "Gefuehlsrueckmeldung erkannt.",
            },
        )
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["es fuehlt sich klar an", "exit"]),
            patch.object(sandbox, "call_semantic_node", return_value=({}, ready_decision)),
            patch.object(sandbox, "_speak_text", return_value=None),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "phase4_common_first_cigarette",
                speak=True,
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("[SCRIPT]", output)
        self.assertIn("Wie fuehlt sich das jetzt fuer dich an?", output)
        self.assertIn("Und spuere jetzt auch, wie sicher du damals vielleicht davon ueberzeugt warst", output)

    def test_run_interactive_same_node_unclear_reply_does_not_reprint_question_immediately(self) -> None:
        unclear_decision = validate_semantic_decision(
            "dark_scene_happening",
            {
                "intent": "unclear",
                "action": "clarify",
                "next_node": "dark_scene_happening",
                "confidence": 1.0,
                "reason": "Unsichere Antwort erkannt.",
            },
        )
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ich bin nicht sicher", "exit"]),
            patch.object(sandbox, "call_semantic_node", return_value=({}, unclear_decision)),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "dark_scene_happening",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.count("Was passiert dort in diesem Moment gerade?"), 1)
        self.assertIn("Wenn sich der Moment klarer zeigt", output)

    def test_run_interactive_reasks_current_question_after_reengagement_signal(self) -> None:
        ready_decision = validate_semantic_decision(
            "dark_scene_people_who",
            {
                "intent": "ready",
                "action": "transition",
                "next_node": "dark_scene_age",
                "confidence": 1.0,
                "reason": "Personen wurden benannt.",
            },
        )
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=[None, "ich bin da", "menschen", "exit"]),
            patch.object(sandbox, "call_semantic_node", return_value=({}, ready_decision)) as mocked_call,
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "dark_scene_people_who",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked_call.call_count, 1)
        self.assertIn("Ich stelle dir die Frage noch einmal", output)
        self.assertEqual(output.count("Kannst du mir sagen, wer oder was dort fuer dich erkennbar wird?"), 2)

    def test_run_interactive_routes_dark_scene_person_hint_to_people_question(self) -> None:
        visual_decision = validate_semantic_decision(
            "dark_scene_perception",
            {
                "intent": "visual",
                "action": "transition",
                "next_node": "dark_scene_who",
                "confidence": 1.0,
                "reason": "Visueller Zugang erkannt.",
            },
        )
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ich sehe eine person", "exit"]),
            patch.object(sandbox, "call_semantic_node", return_value=({}, visual_decision)),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "dark_scene_perception",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("intent=visual action=transition next=dark_scene_people_who", output)
        self.assertIn("Kannst du schon etwas genauer erkennen, wer diese Person sein koennte?", output)
        self.assertNotIn("\n[FRAGE]\nWas siehst du dort genau?", output)

    def test_run_interactive_debug_model_prints_live_trace_events(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ja", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "group_person_ready",
                semantic_provider="local-intent",
                debug_model=True,
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("[TRACE]", output)
        self.assertIn('"stage": "call_start"', output)
        self.assertIn('"stage": "local_session_decision_hit"', output)

    def test_run_interactive_dark_scene_people_who_meta_and_generic_replies_get_identity_specific_followup(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=["ich sehe jemand", "ja", "ich sehe jemand", "ein freund", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "dark_scene_perception",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        normalized_output = " ".join(output.split())
        self.assertEqual(exit_code, 0)
        self.assertIn("Kannst du schon etwas genauer erkennen, wer diese Person sein koennte?", normalized_output)
        self.assertIn(
            "Dann sag mir bitte jetzt direkt, wer diese Person sein koennte, also den Namen oder eine kurze Beschreibung.",
            normalized_output,
        )
        self.assertTrue(
            any(
                snippet in normalized_output
                for snippet in (
                    "Ich brauche hier noch etwas genauer, wer diese Person sein koennte.",
                    "Bleib bei deinem ersten Eindruck: Wer koennte diese Person sein?",
                    "Wenn du noch einen Moment hinschaust: Wird etwas deutlicher, wer diese Person sein koennte?",
                )
            )
        )
        self.assertIn("intent=ready action=transition next=dark_scene_age", normalized_output)
        self.assertIn("Wie alt bist du dort?", normalized_output)

    def test_run_interactive_starts_safe_outro_after_repeated_silence(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=[None, None, None, None, None, None, None, None, None]),
            patch("builtins.input", return_value=""),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "session_phase2_ready",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("beginne ich jetzt eine ruhige Ausleitung", output)
        self.assertIn("status=ended_by_inactivity reason=no_customer_response_safe_outro", output)
        self.assertIn("Und jetzt...", output)

    def test_run_interactive_body_bridge_does_not_jump_back_to_visual_prompt(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=[None, None, None, "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "hell_light_level",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        normalized_output = " ".join(output.split())
        self.assertEqual(exit_code, 0)
        self.assertIn("Dann gehen wir jetzt ohne klares Bild ueber das weiter", output)
        self.assertIn(
            "Wenn du dort nichts klar siehst oder hoerst: Was nimmst du ueber Koerper, Geruch, Geschmack oder Temperatur wahr?",
            normalized_output,
        )
        self.assertNotIn("Und was nimmst du dort sonst noch wahr, siehst du oder hoerst du was?", output)

    def test_run_interactive_body_bridge_reclassifies_late_dark_then_visual_access(self) -> None:
        fake_router = self._FakeLocalIntentRouter('{"intent":"unclear"}')
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", fake_router)),
            patch.object(sandbox, "_timed_input", side_effect=[None, None, None, "dunkel", "ich sehe was", "exit"]),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "hell_light_level",
                semantic_provider="local-intent",
            )
        normalized_output = " ".join(captured.getvalue().split())
        self.assertEqual(exit_code, 0)
        self.assertIn("intent=ready action=transition next=dark_scene_perception", normalized_output)
        self.assertIn("Und was nimmst du dort sonst noch wahr, siehst du oder hoerst du was?", normalized_output)
        self.assertIn("intent=visual action=transition next=dark_scene_who", normalized_output)
        self.assertIn("Was siehst du dort genau?", normalized_output)

    def test_run_interactive_preserves_silence_warning_for_contextual_origin_person_name(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sandbox, "_resolve_semantic_backends", return_value=(None, "", None)),
            patch.object(sandbox, "_timed_input", side_effect=[None, None, None, None, None, None, None, None, None]),
            patch("builtins.input", return_value=""),
            redirect_stdout(captured),
        ):
            exit_code = run_interactive(
                "origin_person_name",
                semantic_provider="local-intent",
            )
        output = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("zusammenhaengenden Rahmen fortsetzen", output)
        self.assertIn("noch immer keine Rueckmeldung", output)
        self.assertIn("beginne ich jetzt eine ruhige Ausleitung", output)

    def test_tts_failure_does_not_abort_runtime(self) -> None:
        previous_ready = sandbox._TTS_READY
        previous_error = sandbox._TTS_ERROR
        sandbox._TTS_READY = None
        sandbox._TTS_ERROR = None
        try:
            with (
                patch.object(sandbox, "_ensure_tts_ready", return_value=True),
                patch.object(sandbox, "_synthesize_google_wav_to_cache", side_effect=RuntimeError("tts kaputt")),
                patch("builtins.print") as mocked_print,
            ):
                sandbox._speak_text("kurzer test")
            self.assertFalse(sandbox._TTS_READY)
            self.assertEqual(sandbox._TTS_ERROR, "tts kaputt")
            mocked_print.assert_any_call("[TTS deaktiviert] tts kaputt")
        finally:
            sandbox._TTS_READY = previous_ready
            sandbox._TTS_ERROR = previous_error

    def test_available_nodes_include_session_and_phase4_nodes(self) -> None:
        self.assertIn("session_phase1_intro", available_node_ids())
        self.assertIn("session_phase1_preflight_check", available_node_ids())
        self.assertIn("session_phase1_anchor_before_focus", available_node_ids())
        self.assertIn("phase4_common_done_signal", available_node_ids())
        self.assertIn("phase4_common_first_cigarette_consequences", available_node_ids())
        self.assertIn("hell_feel_branch", available_node_ids())

    def test_main_blocks_interactive_openai_without_live_api_flag(self) -> None:
        captured = io.StringIO()
        with (
            patch.object(sys, "argv", ["run_session_sandbox.py", "--node", "session_phase1_intro"]),
            patch.object(sandbox, "run_interactive") as run_interactive,
            redirect_stdout(captured),
        ):
            exit_code = sandbox.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("interaktive API-Laeufe standardmaessig blockiert", captured.getvalue())
        run_interactive.assert_not_called()

    def test_main_interactive_openai_default_uses_budget_guard(self) -> None:
        fake_budget = unittest.mock.Mock()
        fake_budget.summary.return_value = "guard ok"
        captured = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_session_sandbox.py",
                    "--node",
                    "session_phase1_intro",
                    "--live-api",
                    "--max-api-calls",
                    "7",
                ],
            ),
            patch.object(sandbox, "build_live_api_budget", return_value=fake_budget) as build_budget,
            patch.object(sandbox, "run_interactive", return_value=0) as run_interactive,
            redirect_stdout(captured),
        ):
            exit_code = sandbox.main()

        self.assertEqual(exit_code, 0)
        build_budget.assert_called_once()
        self.assertEqual(build_budget.call_args.kwargs["estimated_calls"], 7)
        self.assertEqual(build_budget.call_args.kwargs["requested_max_calls"], 7)
        run_interactive.assert_called_once()
        self.assertEqual(run_interactive.call_args.kwargs["semantic_provider"], "openai-router")
        self.assertIs(run_interactive.call_args.kwargs["live_api_budget"], fake_budget)
        self.assertIn("guard ok", captured.getvalue())

    def test_main_local_intent_interactive_bypasses_live_guard(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_session_sandbox.py",
                    "--node",
                    "session_phase1_intro",
                    "--semantic-provider",
                    "local-intent",
                ],
            ),
            patch.object(sandbox, "build_live_api_budget") as build_budget,
            patch.object(sandbox, "run_interactive", return_value=0) as run_interactive,
        ):
            exit_code = sandbox.main()

        self.assertEqual(exit_code, 0)
        build_budget.assert_not_called()
        run_interactive.assert_called_once()
        self.assertEqual(run_interactive.call_args.kwargs["semantic_provider"], "local-intent")
        self.assertIsNone(run_interactive.call_args.kwargs["live_api_budget"])

    def test_global_question_matrix_hits_question_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for node_id in semantic_node_ids:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "wie meinst du das")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "question")

    def test_global_fatigue_matrix_hits_support_needed_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for node_id in semantic_node_ids:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "ich schlafe fast ein")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "support_needed")

    def test_global_ambivalence_matrix_hits_support_needed_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for node_id in semantic_node_ids:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "ich weiss nicht ob ich das will")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "support_needed")

    def test_global_hostile_matrix_hits_support_needed_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for node_id in semantic_node_ids:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "du bist ein idiot")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "support_needed")

    def test_global_nonanswer_noise_matrix_hits_unclear_or_support_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for phrase in ("asdf", "...", "hm", "ich bin ein toaster"):
            for node_id in semantic_node_ids:
                with self.subTest(node_id=node_id, phrase=phrase):
                    decision = _local_session_decision(node_id, phrase)
                    self.assertIsNotNone(decision)
                    assert decision is not None
                    self.assertIn(decision.intent, {"unclear", "support_needed"})

    def test_call_semantic_node_keeps_obvious_noise_local_without_router_call(self) -> None:
        for node_id, phrase in (
            ("dark_scene_who", "asdf"),
            ("dark_scene_audio_detail", "..."),
            ("origin_trigger_source", "hm"),
            ("group_person_trigger_reason", "ich bin ein toaster"),
        ):
            with self.subTest(node_id=node_id, phrase=phrase):
                fake_router = self._FakeLocalIntentRouter('{"intent":"ready"}')
                parsed, decision = call_semantic_node(
                    None,
                    "",
                    node_id,
                    phrase,
                    runtime_slots={"named_person": "diese Person"},
                    local_intent_router=fake_router,
                )
                self.assertEqual(fake_router.calls, 0)
                self.assertEqual(parsed["source"], "local_parser")
                self.assertIn(decision.intent, {"unclear", "support_needed"})
                self.assertEqual(decision.next_node, node_id)

    def test_global_abort_matrix_hits_abort_for_all_semantic_nodes(self) -> None:
        semantic_node_ids = [
            node_id for node_id in available_node_ids() if not isinstance(get_node_spec(node_id), ScriptNodeSpec)
        ]
        for node_id in semantic_node_ids:
            with self.subTest(node_id=node_id):
                decision = _local_session_decision(node_id, "abbrechen")
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(decision.intent, "abort")


if __name__ == "__main__":
    unittest.main()
