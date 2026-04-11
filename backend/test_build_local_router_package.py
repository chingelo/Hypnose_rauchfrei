import json
import unittest
from pathlib import Path

from build_local_router_package import build_local_router_package, write_local_router_package
from build_gold_v3_datasets import write_v3_artifacts


class BuildLocalRouterPackageTests(unittest.TestCase):
    def test_build_local_router_package_contains_expected_counts(self) -> None:
        package = build_local_router_package()
        self.assertEqual(len(package["routing_train_only.jsonl"]), 355)
        self.assertEqual(len(package["routing_eval_only.jsonl"]), 73)
        self.assertEqual(len(package["slot_train_only.jsonl"]), 68)
        self.assertEqual(len(package["slot_eval_only.jsonl"]), 16)
        self.assertEqual(len(package["router_sft_train.jsonl"]), 423)
        self.assertEqual(len(package["router_sft_eval.jsonl"]), 89)

    def test_examples_are_chat_formatted(self) -> None:
        package = build_local_router_package()
        example = package["router_sft_train.jsonl"][0]
        self.assertIn("messages", example)
        self.assertEqual(example["messages"][0]["role"], "system")
        self.assertEqual(example["messages"][1]["role"], "user")
        self.assertEqual(example["messages"][2]["role"], "assistant")
        self.assertIn("allowed_intents", example["messages"][1]["content"])

    def test_intent_only_package_uses_intent_only_assistant_output(self) -> None:
        package = build_local_router_package(routing_mode="intent_only")
        example = package["routing_train_only.jsonl"][0]
        assistant = json.loads(example["messages"][2]["content"])
        self.assertEqual(set(assistant.keys()), {"intent"})
        self.assertIn("Nur das Feld `intent`", example["messages"][1]["content"])

    def test_write_local_router_package_writes_manifest(self) -> None:
        manifest = write_local_router_package()
        path = Path("finetune_data") / "v2" / "local_router" / "router_package_manifest.json"
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["package"], "local_router")
        self.assertEqual(loaded["counts"]["router_sft_eval.jsonl"], 89)
        self.assertEqual(manifest["counts"]["router_sft_train.jsonl"], 423)

    def test_write_local_router_package_supports_v3(self) -> None:
        write_v3_artifacts()
        manifest = write_local_router_package(dataset_version="v3")
        path = Path("finetune_data") / "v3" / "local_router" / "router_package_manifest.json"
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["dataset_version"], "v3")
        self.assertEqual(loaded["counts"], manifest["counts"])

    def test_write_local_router_package_supports_intent_only(self) -> None:
        manifest = write_local_router_package(dataset_version="v3", routing_mode="intent_only")
        path = Path("finetune_data") / "v3" / "local_router_intent" / "router_package_manifest.json"
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["routing_mode"], "intent_only")
        self.assertEqual(loaded["package"], "local_router_intent")
        self.assertEqual(loaded["counts"], manifest["counts"])


if __name__ == "__main__":
    unittest.main()
