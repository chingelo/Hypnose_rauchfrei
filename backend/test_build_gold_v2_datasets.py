import json
import unittest
from pathlib import Path

from build_gold_v2_datasets import (
    SLOT_EDGE_CASES,
    ROUTING_EDGE_CASES,
    build_routing_v2,
    build_slot_v2,
    build_v2_splits,
    write_v2_artifacts,
)


class BuildGoldV2DatasetsTests(unittest.TestCase):
    def test_routing_v2_adds_edge_cases(self) -> None:
        rows, review = build_routing_v2()
        self.assertGreaterEqual(review["v2_count"], review["v1_count"] + 10)
        labels = {row.get("label") for row in rows if row["source"][0] == "gold_v2_edge_cases"}
        self.assertIn("group_core_real_explanation", labels)
        self.assertIn("origin_kind_group_smoking_clique", labels)
        self.assertGreaterEqual(review["synthetic_marker_counts_v1"]["placeholder_banane"], 40)

    def test_slot_v2_exposes_helper_mismatches(self) -> None:
        rows, review = build_slot_v2()
        self.assertGreaterEqual(review["v2_count"], review["v1_count"] + 10)
        self.assertEqual(review["helper_alignment_mismatches"], 0)
        self.assertEqual(review["helper_alignment_examples"], [])

    def test_v2_splits_keep_total_counts(self) -> None:
        routing_rows, _ = build_routing_v2()
        split = build_v2_splits(routing_rows, dataset="routing")
        self.assertEqual(len(split["train"]) + len(split["eval"]), len(routing_rows))
        self.assertGreater(len(split["eval"]), 50)

    def test_write_v2_artifacts_writes_review_file(self) -> None:
        review = write_v2_artifacts()
        review_path = Path("finetune_data") / "v2" / "gold_v2_review.json"
        self.assertTrue(review_path.exists())
        loaded = json.loads(review_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["version"], 2)
        self.assertIn("routing_eval_v2.jsonl", loaded["split_counts"])


if __name__ == "__main__":
    unittest.main()
