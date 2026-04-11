import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from build_gold_v3_datasets import write_v3_artifacts
from evaluate_finetune_candidate import (
    ReferenceProvider,
    _build_prompt,
    _compare_output,
    _dataset_dir,
    _evaluate_dataset,
    _normalize_text,
    _report_path,
    _schema_for_dataset,
)


class EvaluateFinetuneCandidateTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(_normalize_text("  a \n  b\tc  "), "a b c")

    def test_compare_output_uses_text_normalization_for_reply_tasks(self) -> None:
        exact, similarity = _compare_output(
            "clarification",
            {"reply_text": "Hallo   Welt"},
            {"reply_text": "Hallo Welt"},
        )
        self.assertTrue(exact)
        self.assertGreaterEqual(similarity or 0.0, 0.99)

    def test_build_prompt_contains_schema_and_input(self) -> None:
        schema = _schema_for_dataset("routing")
        row = {
            "id": "routing-test",
            "input": {
                "node_id": "dark_known_branch",
                "question_text": "Kennst du das?",
                "runtime_slots": {},
                "session_hint": "",
                "user_reply": "zum ersten Mal",
            },
            "output": {"intent": "new", "action": "transition", "next_node": "dark_origin_terminal"},
        }
        prompt = _build_prompt("routing", row, schema)
        self.assertIn('"task": "routing"', prompt)
        self.assertIn('"required_schema"', prompt)
        self.assertIn('"node_id": "dark_known_branch"', prompt)

    def test_build_prompt_supports_routing_intent_only(self) -> None:
        row = {
            "id": "routing-test",
            "input": {
                "node_id": "dark_known_branch",
                "question_text": "Kennst du das?",
                "runtime_slots": {},
                "session_hint": "",
                "user_reply": "zum ersten Mal",
            },
            "output": {"intent": "new", "action": "transition", "next_node": "dark_origin_terminal"},
        }
        schema = {
            "type": "object",
            "required": ["intent"],
            "properties": {"intent": {"type": "string", "enum": ["known", "new"]}},
        }
        prompt = _build_prompt("routing", row, schema, routing_mode="intent_only")
        self.assertIn('"task": "routing_intent"', prompt)
        self.assertIn('"enum": [', prompt)
        self.assertIn('Nur das Feld `intent`', prompt)

    def test_reference_provider_evaluation_is_perfect(self) -> None:
        metrics, results = _evaluate_dataset("routing", ReferenceProvider(), version="v1", limit=5)
        self.assertEqual(metrics.total, 5)
        self.assertEqual(metrics.json_parse_success, 5)
        self.assertEqual(metrics.schema_valid, 5)
        self.assertEqual(metrics.exact_match, 5)
        self.assertTrue(all(result.error is None for result in results))

    def test_dataset_dir_supports_v2(self) -> None:
        self.assertTrue((_dataset_dir("v2") / "routing_eval_v2.jsonl").exists())

    def test_dataset_dir_supports_v3(self) -> None:
        write_v3_artifacts()
        self.assertTrue((_dataset_dir("v3") / "routing_eval_v3.jsonl").exists())

    def test_report_path_uses_local_adapter_suffix(self) -> None:
        args = Namespace(
            report_path=None,
            provider="local_adapter",
            adapter_dir=r"C:\tmp\router_adapter",
            model=None,
            dataset_version="v2",
            routing_mode="full",
        )
        report_path = _report_path(args)
        self.assertTrue(report_path.name.endswith("local_adapter_router_adapter.json"))

    def test_compare_output_supports_routing_intent_only(self) -> None:
        row = {
            "input": {"node_id": "dark_known_branch"},
        }
        exact, similarity = _compare_output(
            "routing",
            {"intent": "new", "action": "transition", "next_node": "dark_origin_terminal"},
            {"intent": "new"},
            row=row,
            routing_mode="intent_only",
        )
        self.assertTrue(exact)
        self.assertIsNone(similarity)


if __name__ == "__main__":
    unittest.main()
