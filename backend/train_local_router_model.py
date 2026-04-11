from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT_DIR / "finetune_data" / "v2" / "local_router"
DEFAULT_MANIFEST_PATH = DEFAULT_PACKAGE_DIR / "router_package_manifest.json"
DEFAULT_CONFIG_PATH = DEFAULT_PACKAGE_DIR / "router_qlora_config.json"

REQUIRED_ML_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
]

OPTIONAL_ML_MODULES = [
    "bitsandbytes",
]


def _log(message: str) -> None:
    print(f"[local-router-train] {message}", flush=True)


def _base_model_available_offline(base_model: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(base_model, "config.json", local_files_only=True)
        hf_hub_download(base_model, "model.safetensors.index.json", local_files_only=True)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DependencyReport:
    available: dict[str, bool]

    @property
    def missing_required(self) -> list[str]:
        return [name for name in REQUIRED_ML_MODULES if not self.available.get(name, False)]

    @property
    def missing_optional(self) -> list[str]:
        return [name for name in OPTIONAL_ML_MODULES if not self.available.get(name, False)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_package_dir(manifest_path: Path, config_path: Path) -> Path:
    manifest_dir = manifest_path.resolve().parent
    config_dir = config_path.resolve().parent
    if manifest_dir != config_dir:
        raise ValueError(
            "Manifest und Config muessen im selben Package-Verzeichnis liegen: "
            f"{manifest_dir} != {config_dir}"
        )
    return manifest_dir


def probe_dependencies() -> DependencyReport:
    available: dict[str, bool] = {}
    for module_name in REQUIRED_ML_MODULES + OPTIONAL_ML_MODULES:
        try:
            importlib.import_module(module_name)
            available[module_name] = True
        except Exception:
            available[module_name] = False
    return DependencyReport(available=available)


def load_router_package(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    package_dir = _resolve_package_dir(manifest_path, config_path)
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    train_rows = _load_jsonl(package_dir / "router_sft_train.jsonl")
    eval_rows = _load_jsonl(package_dir / "router_sft_eval.jsonl")
    return {
        "package_dir": str(package_dir),
        "manifest": manifest,
        "config": config,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def build_training_plan(
    *,
    base_model: str,
    output_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    package = load_router_package(manifest_path=manifest_path, config_path=config_path)
    dependencies = probe_dependencies()
    return {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "package_dir": package["package_dir"],
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "train_count": len(package["train_rows"]),
        "eval_count": len(package["eval_rows"]),
        "tasks": package["manifest"]["scope"],
        "excluded_scope": package["manifest"]["excluded_scope"],
        "missing_required_modules": dependencies.missing_required,
        "missing_optional_modules": dependencies.missing_optional,
        "config": package["config"],
    }


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message["role"].strip().upper()
        content = message["content"].strip()
        lines.append(f"<|{role}|>\n{content}")
    return "\n\n".join(lines)


def _require_training_dependencies() -> None:
    report = probe_dependencies()
    if report.missing_required:
        missing = ", ".join(report.missing_required)
        raise RuntimeError(
            "Fehlende Trainings-Dependencies: "
            f"{missing}. Installiere zuerst backend/requirements_local_router_train.txt."
        )


def run_training(
    *,
    base_model: str,
    output_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    _require_training_dependencies()
    _log(f"Training vorbereitet mit Base-Modell: {base_model}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    package = load_router_package(manifest_path=manifest_path, config_path=config_path)
    config = package["config"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output-Verzeichnis: {output_dir}")
    local_files_only = _base_model_available_offline(base_model)
    _log(f"Lokaler Modell-Cache verfügbar: {'ja' if local_files_only else 'nein'}")

    train_dataset = Dataset.from_list(
        [
            {
                "text": _messages_to_text(row["messages"]),
            }
            for row in package["train_rows"]
        ]
    )
    eval_dataset = Dataset.from_list(
        [
            {
                "text": _messages_to_text(row["messages"]),
            }
            for row in package["eval_rows"]
        ]
    )
    _log(
        "Datasets geladen: "
        f"train={len(train_dataset)}, eval={len(eval_dataset)}, "
        f"tasks={', '.join(package['manifest']['scope'])}"
    )

    _log("Tokenizer wird geladen.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception:
        _log("Fast Tokenizer fehlgeschlagen, fallback auf use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=False,
            local_files_only=local_files_only,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _log("Tokenizer geladen.")

    quantization_config = None
    try:
        BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        quantization_config = None
    _log(
        "Quantisierung: "
        + ("4bit QLoRA aktiv." if quantization_config is not None else "keine Quantisierung.")
    )

    _log("Basismodell wird geladen. Dieser Schritt kann mehrere Minuten dauern.")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        local_files_only=local_files_only,
    )
    _log("Basismodell geladen.")

    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    _log("LoRA-Adapter auf Modell angewendet.")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=True,
        dataset_text_field="text",
        max_seq_length=int(config["max_seq_length"]),
        packing=bool(config["packing"]),
    )
    _log("TrainingArguments erstellt.")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    _log("Trainer erstellt. Trainingslauf startet jetzt.")
    trainer.train()
    _log("Training abgeschlossen. Modell wird gespeichert.")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _log("Modell und Tokenizer gespeichert.")
    return output_dir


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or train the local structured router model.")
    parser.add_argument("--base-model", required=True, help="HF base model id for QLoRA training.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the trained adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate package and print the training plan.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Override package manifest path.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Override QLoRA config path.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    config_path = Path(args.config_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = manifest_path.resolve().parent / "artifacts" / "router_qlora"

    plan = build_training_plan(
        base_model=args.base_model,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    _print_plan(plan)

    if args.dry_run:
        return 0

    if plan["missing_required_modules"]:
        missing = ", ".join(plan["missing_required_modules"])
        print(
            "\nTraining nicht gestartet. Fehlende Module: "
            f"{missing}\nInstalliere zuerst backend/requirements_local_router_train.txt."
        )
        return 1

    result_dir = run_training(
        base_model=args.base_model,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    print(f"\nTraining abgeschlossen. Artefakte: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
