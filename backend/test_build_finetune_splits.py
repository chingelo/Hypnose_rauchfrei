import unittest

from build_finetune_splits import build_schemas, build_splits


class BuildFinetuneSplitsTests(unittest.TestCase):
    def test_build_splits_produces_expected_files(self) -> None:
        splits = build_splits()
        self.assertEqual(
            set(splits.keys()),
            {
                "routing_train.jsonl",
                "routing_eval.jsonl",
                "slot_extraction_train.jsonl",
                "slot_extraction_eval.jsonl",
                "clarification_train.jsonl",
                "clarification_eval.jsonl",
                "support_abort_train.jsonl",
                "support_abort_eval.jsonl",
            },
        )

    def test_splits_are_non_empty_and_disjoint(self) -> None:
        splits = build_splits()
        for prefix in ("routing", "slot_extraction", "clarification", "support_abort"):
            train = splits[f"{prefix}_train.jsonl"]
            eval_rows = splits[f"{prefix}_eval.jsonl"]
            self.assertGreater(len(train), 0)
            self.assertGreater(len(eval_rows), 0)
            train_ids = {row["id"] for row in train}
            eval_ids = {row["id"] for row in eval_rows}
            self.assertTrue(train_ids.isdisjoint(eval_ids))

    def test_split_sizes_match_current_gold_totals(self) -> None:
        splits = build_splits()
        self.assertEqual(len(splits["routing_train.jsonl"]) + len(splits["routing_eval.jsonl"]), 403)
        self.assertEqual(
            len(splits["slot_extraction_train.jsonl"]) + len(splits["slot_extraction_eval.jsonl"]),
            59,
        )
        self.assertEqual(
            len(splits["clarification_train.jsonl"]) + len(splits["clarification_eval.jsonl"]),
            295,
        )
        self.assertEqual(
            len(splits["support_abort_train.jsonl"]) + len(splits["support_abort_eval.jsonl"]),
            125,
        )

    def test_routing_eval_contains_multiple_nodes(self) -> None:
        splits = build_splits()
        eval_nodes = {row["input"]["node_id"] for row in splits["routing_eval.jsonl"]}
        self.assertGreaterEqual(len(eval_nodes), 20)

    def test_clarification_and_support_eval_sizes_are_meaningful(self) -> None:
        splits = build_splits()
        self.assertGreaterEqual(len(splits["clarification_eval.jsonl"]), 40)
        self.assertGreaterEqual(len(splits["support_abort_eval.jsonl"]), 20)

    def test_schemas_are_strict_and_task_specific(self) -> None:
        schemas = build_schemas()
        self.assertIn("intent", schemas["routing_output.schema.json"]["required"])
        self.assertIn("reply_text", schemas["clarification_output.schema.json"]["required"])
        self.assertFalse(schemas["support_abort_output.schema.json"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
