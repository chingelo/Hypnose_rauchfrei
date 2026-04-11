import json
import unittest
from pathlib import Path

from train_local_router_model import build_training_plan, load_router_package, probe_dependencies


class TrainLocalRouterModelTests(unittest.TestCase):
    def test_load_router_package_counts(self) -> None:
        package = load_router_package()
        self.assertEqual(len(package["train_rows"]), 423)
        self.assertEqual(len(package["eval_rows"]), 89)
        self.assertIn("routing", package["manifest"]["scope"])
        self.assertIn("slot_extraction", package["manifest"]["scope"])

    def test_load_router_package_v3_counts(self) -> None:
        manifest_path = Path("finetune_data") / "v3" / "local_router" / "router_package_manifest.json"
        config_path = Path("finetune_data") / "v3" / "local_router" / "router_qlora_config.json"
        package = load_router_package(manifest_path=manifest_path, config_path=config_path)
        self.assertEqual(len(package["train_rows"]), 456)
        self.assertEqual(len(package["eval_rows"]), 99)
        self.assertEqual(package["manifest"]["dataset_version"], "v3")

    def test_build_training_plan_contains_dependency_report(self) -> None:
        plan = build_training_plan(
            base_model="dummy/base-model",
            output_dir=Path("artifacts") / "router_qlora",
        )
        self.assertEqual(plan["train_count"], 423)
        self.assertEqual(plan["eval_count"], 89)
        self.assertIn("missing_required_modules", plan)
        self.assertIn("config", plan)
        self.assertTrue(plan["package_dir"].endswith(r"finetune_data\v2\local_router"))

    def test_build_training_plan_v3_package(self) -> None:
        manifest_path = Path("finetune_data") / "v3" / "local_router" / "router_package_manifest.json"
        config_path = Path("finetune_data") / "v3" / "local_router" / "router_qlora_config.json"
        plan = build_training_plan(
            base_model="dummy/base-model",
            output_dir=Path("artifacts") / "router_qlora_v3",
            manifest_path=manifest_path,
            config_path=config_path,
        )
        self.assertEqual(plan["train_count"], 456)
        self.assertEqual(plan["eval_count"], 99)
        self.assertTrue(plan["package_dir"].endswith(r"finetune_data\v3\local_router"))

    def test_probe_dependencies_reports_known_modules(self) -> None:
        report = probe_dependencies()
        self.assertIn("transformers", report.available)
        self.assertIn("torch", report.available)


if __name__ == "__main__":
    unittest.main()
