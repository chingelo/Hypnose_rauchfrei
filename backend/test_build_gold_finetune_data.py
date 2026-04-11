import json
import tempfile
import unittest
from pathlib import Path

from build_gold_finetune_data import (
    OUTPUT_DIR,
    build_all_datasets,
)


class BuildGoldFinetuneDataTests(unittest.TestCase):
    def test_build_all_datasets_returns_expected_keys(self) -> None:
        datasets = build_all_datasets()
        self.assertEqual(
            set(datasets.keys()),
            {
                "routing_gold.jsonl",
                "slot_extraction_gold.jsonl",
                "clarification_gold.jsonl",
                "support_abort_gold.jsonl",
            },
        )

    def test_dataset_sizes_are_nontrivial(self) -> None:
        datasets = build_all_datasets()
        self.assertGreaterEqual(len(datasets["routing_gold.jsonl"]), 250)
        self.assertGreaterEqual(len(datasets["slot_extraction_gold.jsonl"]), 40)
        self.assertGreaterEqual(len(datasets["clarification_gold.jsonl"]), 150)
        self.assertGreaterEqual(len(datasets["support_abort_gold.jsonl"]), 100)

    def test_routing_dataset_contains_group_branch_example(self) -> None:
        datasets = build_all_datasets()
        self.assertTrue(
            any(
                record["input"]["node_id"] == "origin_other_target_kind"
                and record["input"]["user_reply"] == "eine Gruppe"
                and record["output"]["next_node"] == "group_branch_intro"
                for record in datasets["routing_gold.jsonl"]
            )
        )

    def test_slot_extraction_dataset_contains_group_other_and_person(self) -> None:
        datasets = build_all_datasets()
        outputs = {
            (record["task"], record["input"].get("user_reply") or record["input"].get("trigger_focus_ref")): record["output"]
            for record in datasets["slot_extraction_gold.jsonl"]
        }
        self.assertEqual(outputs[("focus_kind", "die menschen wo rauchen")]["focus_kind"], "group")
        self.assertEqual(outputs[("focus_kind", "der geruch von rauch")]["focus_kind"], "other")
        self.assertEqual(outputs[("named_person_label", "mein vater")]["named_person"], "Mein Vater")

    def test_clarification_dataset_contains_empty_input_reply_for_scene_access(self) -> None:
        datasets = build_all_datasets()
        self.assertTrue(
            any(
                record["task"] == "empty_input_reply"
                and record["input"]["node_id"] == "scene_access_followup"
                and "Falls du gerade nichts erkennen kannst" in record["output"]["reply_text"]
                for record in datasets["clarification_gold.jsonl"]
            )
        )

    def test_support_abort_dataset_contains_inactivity_texts(self) -> None:
        datasets = build_all_datasets()
        replies = {record["output"]["reply_text"] for record in datasets["support_abort_gold.jsonl"]}
        self.assertTrue(any("keine Rueckmeldung" in reply for reply in replies))
        self.assertTrue(any("beende ich die Sitzung jetzt" in reply for reply in replies))

    def test_written_jsonl_is_ascii_json_per_line(self) -> None:
        datasets = build_all_datasets()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for name, records in datasets.items():
                path = temp_root / name
                with path.open("w", encoding="utf-8", newline="\n") as handle:
                    for record in records[:3]:
                        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertGreaterEqual(len(lines), 3)
                for line in lines:
                    parsed = json.loads(line)
                    self.assertIn("id", parsed)
                    self.assertIn("input", parsed)
                    self.assertIn("output", parsed)


if __name__ == "__main__":
    unittest.main()
