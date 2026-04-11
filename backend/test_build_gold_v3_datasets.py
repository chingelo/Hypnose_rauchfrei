import json
import unittest
from pathlib import Path

from build_gold_v3_datasets import build_routing_v3, build_slot_v3, build_v3_splits, write_v3_artifacts


class BuildGoldV3DatasetsTests(unittest.TestCase):
    def test_routing_v3_adds_canonical_edge_cases(self) -> None:
        rows, review = build_routing_v3()
        self.assertGreaterEqual(review["v3_count"], review["v2_count"] + 10)
        v3_labels = {row.get("label") for row in rows if row["source"][0] == "gold_v3_edge_cases"}
        self.assertIn("known_branch_new_plain", v3_labels)
        self.assertIn("hell_wait_resolved", v3_labels)

    def test_slot_v3_can_expose_future_helper_drift(self) -> None:
        rows, review = build_slot_v3()
        self.assertGreaterEqual(review["v3_count"], review["v2_count"] + 5)
        self.assertGreaterEqual(review["helper_alignment_total"], 10)

    def test_v3_splits_keep_total_counts(self) -> None:
        routing_rows, _ = build_routing_v3()
        split = build_v3_splits(routing_rows, dataset="routing")
        self.assertEqual(len(split["train"]) + len(split["eval"]), len(routing_rows))
        self.assertGreater(len(split["eval"]), 50)

    def test_write_v3_artifacts_writes_review_file(self) -> None:
        review = write_v3_artifacts()
        review_path = Path("finetune_data") / "v3" / "gold_v3_review.json"
        self.assertTrue(review_path.exists())
        loaded = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["version"], 3)
        self.assertIn("routing_eval_v3.jsonl", loaded["split_counts"])
        self.assertEqual(loaded["split_counts"], review["split_counts"])


if __name__ == "__main__":
    unittest.main()
