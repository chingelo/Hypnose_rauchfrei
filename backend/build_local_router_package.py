from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_finetune_splits import SCHEMA_DIR
from session_sandbox_orchestrator import SemanticNodeSpec, get_node_spec


ROOT_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ROOT_DIR / "finetune_data"
ROUTING_MODES = ("full", "intent_only")

SCHEMA_FILES = {
    "routing": SCHEMA_DIR / "routing_output.schema.json",
    "slot_extraction": SCHEMA_DIR / "slot_extraction_output.schema.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dataset_dir(dataset_version: str) -> Path:
    if dataset_version == "v2":
        return FINETUNE_DIR / "v2"
    if dataset_version == "v3":
        return FINETUNE_DIR / "v3"
    raise ValueError(dataset_version)


def _package_dir(dataset_version: str) -> Path:
    return _dataset_dir(dataset_version) / "local_router"


def _package_name(routing_mode: str) -> str:
    if routing_mode == "full":
        return "local_router"
    if routing_mode == "intent_only":
        return "local_router_intent"
    raise ValueError(routing_mode)


def _package_dir_for_mode(dataset_version: str, routing_mode: str) -> Path:
    return _dataset_dir(dataset_version) / _package_name(routing_mode)


def _input_files(dataset_version: str) -> dict[str, Path]:
    dataset_dir = _dataset_dir(dataset_version)
    suffix = dataset_version
    return {
        "routing_train": dataset_dir / f"routing_train_{suffix}.jsonl",
        "routing_eval": dataset_dir / f"routing_eval_{suffix}.jsonl",
        "slot_train": dataset_dir / f"slot_extraction_train_{suffix}.jsonl",
        "slot_eval": dataset_dir / f"slot_extraction_eval_{suffix}.jsonl",
    }


def _routing_system_prompt() -> str:
    return (
        "Du bist ein strukturiertes Routing-Modell fuer eine deterministische Hypnose-Runtime. "
        "Klassifiziere die Nutzerantwort nur im Rahmen des aktuellen Knotens. "
        "Gib ausschliesslich ein JSON-Objekt mit intent, action und next_node zurueck. "
        "Keine Erklaerungen, kein Markdown, kein Freitext. "
        "Nutze nur kanonische Labels aus den erlaubten Routen dieses Knotens und erfinde keine Synonyme wie "
        "`first_time`, `see`, `hear`, `smell`, `reaction`, `druck` oder `unknown`."
    )


def _routing_intent_system_prompt() -> str:
    return (
        "Du bist ein strukturiertes Intent-Klassifikationsmodell fuer eine deterministische Hypnose-Runtime. "
        "Klassifiziere die Nutzerantwort nur im Rahmen des aktuellen Knotens. "
        "Gib ausschliesslich ein JSON-Objekt mit dem Feld `intent` zurueck. "
        "Keine Erklaerungen, kein Markdown, kein Freitext. "
        "Nutze nur einen der explizit erlaubten Intents dieses Knotens."
    )


def _slot_system_prompt() -> str:
    return (
        "Du bist ein strukturiertes Extraktionsmodell fuer eine deterministische Hypnose-Runtime. "
        "Extrahiere nur die benoetigten Felder fuer die aktuelle Aufgabe und gib ausschliesslich ein JSON-Objekt zurueck. "
        "Keine Erklaerungen, kein Markdown, kein Freitext. "
        "Bei Gruppen muss `display_trigger_focus_ref` sauber zu `diese Gruppe` normalisiert werden. "
        "Personennamen werden korrekt kapitalisiert."
    )


def _routing_schema(*, allowed_intents: list[str], routing_mode: str) -> dict[str, Any]:
    if routing_mode == "full":
        return _load_json(SCHEMA_FILES["routing"])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RoutingIntentOutput",
        "type": "object",
        "additionalProperties": False,
        "required": ["intent"],
        "properties": {
            "intent": {
                "type": "string",
                "enum": allowed_intents,
            }
        },
    }


def _routing_user_prompt(row: dict[str, Any], schema: dict[str, Any], *, routing_mode: str) -> str:
    node_id = str(row["input"]["node_id"])
    spec = get_node_spec(node_id)
    if not isinstance(spec, SemanticNodeSpec):
        raise TypeError(node_id)
    allowed_routes = {
        intent: {"action": route["action"], "next_node": route["next_node"]}
        for intent, route in spec.routing_rules.items()
    }
    payload = {
        "task": "routing",
        "example_id": row["id"],
        "input": row["input"],
        "required_schema": schema,
        "allowed_intents": list(spec.routing_rules.keys()),
        "rules": [
            "Nur ein JSON-Objekt ausgeben.",
            "Nur innerhalb des aktuellen Knotens entscheiden.",
            "Keine zusaetzlichen Felder ausgeben.",
            "intent muss exakt einer der allowed_intents sein.",
            "Keine synonymen oder freien Labels erfinden.",
        ],
    }
    if routing_mode == "full":
        payload["task"] = "routing"
        payload["allowed_routes"] = allowed_routes
    else:
        payload["task"] = "routing_intent"
        payload["rules"].append("Nur das Feld `intent` ausgeben, nicht `action` oder `next_node`.")
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _slot_user_prompt(row: dict[str, Any], schema: dict[str, Any]) -> str:
    allowed_values: dict[str, Any] | None = None
    if row["task"] == "focus_kind":
        allowed_values = {"focus_kind": ["person", "group", "other"]}
    elif row["task"] == "named_person_label":
        allowed_values = {"named_person": "string|null"}
    elif row["task"] == "display_trigger_focus_ref":
        allowed_values = {"display_trigger_focus_ref": "string"}
    payload = {
        "task": "slot_extraction",
        "example_id": row["id"],
        "input": row["input"],
        "required_schema": schema,
        "allowed_values": allowed_values,
        "rules": [
            "Nur ein JSON-Objekt ausgeben.",
            "Nur die benoetigten Felder aus dem Schema verwenden.",
            "Keine Erklaerungen oder Zusatztexte ausgeben.",
            "Gruppenbezeichnungen werden fuer die Anzeige zu `diese Gruppe` normalisiert.",
        ],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _to_chat_example(
    row: dict[str, Any],
    *,
    task: str,
    schema: dict[str, Any],
    routing_mode: str = "full",
) -> dict[str, Any]:
    if task == "routing":
        system = _routing_system_prompt() if routing_mode == "full" else _routing_intent_system_prompt()
        user = _routing_user_prompt(row, schema, routing_mode=routing_mode)
        assistant_output = (
            row["output"]
            if routing_mode == "full"
            else {"intent": row["output"]["intent"]}
        )
    elif task == "slot_extraction":
        system = _slot_system_prompt()
        user = _slot_user_prompt(row, schema)
        assistant_output = row["output"]
    else:
        raise ValueError(task)

    return {
        "id": row["id"],
        "task": task,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=True)},
        ],
        "metadata": {
            "source": row.get("source", []),
            "input": row["input"],
            "output": row["output"],
            "routing_mode": routing_mode if task == "routing" else None,
        },
    }


def build_local_router_package(
    *,
    dataset_version: str = "v2",
    routing_mode: str = "full",
) -> dict[str, list[dict[str, Any]]]:
    slot_schema = _load_json(SCHEMA_FILES["slot_extraction"])
    input_files = _input_files(dataset_version)

    routing_train = [
        _to_chat_example(
            row,
            task="routing",
            schema=_routing_schema(
                allowed_intents=list(get_node_spec(str(row["input"]["node_id"])).routing_rules.keys()),
                routing_mode=routing_mode,
            ),
            routing_mode=routing_mode,
        )
        for row in _load_jsonl(input_files["routing_train"])
    ]
    routing_eval = [
        _to_chat_example(
            row,
            task="routing",
            schema=_routing_schema(
                allowed_intents=list(get_node_spec(str(row["input"]["node_id"])).routing_rules.keys()),
                routing_mode=routing_mode,
            ),
            routing_mode=routing_mode,
        )
        for row in _load_jsonl(input_files["routing_eval"])
    ]
    slot_train = [
        _to_chat_example(row, task="slot_extraction", schema=slot_schema)
        for row in _load_jsonl(input_files["slot_train"])
    ]
    slot_eval = [
        _to_chat_example(row, task="slot_extraction", schema=slot_schema)
        for row in _load_jsonl(input_files["slot_eval"])
    ]

    train = routing_train + slot_train
    eval_rows = routing_eval + slot_eval
    return {
        "router_sft_train.jsonl": train,
        "router_sft_eval.jsonl": eval_rows,
        "routing_train_only.jsonl": routing_train,
        "routing_eval_only.jsonl": routing_eval,
        "slot_train_only.jsonl": slot_train,
        "slot_eval_only.jsonl": slot_eval,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _qlora_config() -> dict[str, Any]:
    return {
        "version": 1,
        "purpose": "local_structured_router_model",
        "recommended_base_model_class": "7B_instruct_chat_model",
        "training_method": "qlora_sft",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_length": 2048,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "packing": False,
        "notes": [
            "Nur routing + slot_extraction trainieren; clarification/support bleiben deterministisch im Code.",
            "JSON-Ausgabe strikt halten.",
            "Eval immer gegen router_sft_eval.jsonl fahren.",
        ],
    }


def write_local_router_package(*, dataset_version: str = "v2", routing_mode: str = "full") -> dict[str, Any]:
    package_dir = _package_dir_for_mode(dataset_version, routing_mode)
    package_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_local_router_package(dataset_version=dataset_version, routing_mode=routing_mode)
    for name, rows in datasets.items():
        _write_jsonl(package_dir / name, rows)

    if routing_mode == "full":
        routing_schema_path = SCHEMA_FILES["routing"]
    else:
        routing_schema_path = package_dir / "routing_intent_output.schema.json"
        routing_schema_path.write_text(
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "RoutingIntentOutput",
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["intent"],
                    "properties": {
                        "intent": {"type": "string", "minLength": 1},
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "version": 1,
        "package": _package_name(routing_mode),
        "dataset_version": dataset_version,
        "routing_mode": routing_mode,
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "schemas": {
            "routing": str(routing_schema_path),
            "slot_extraction": str(SCHEMA_FILES["slot_extraction"]),
        },
        "scope": [
            "routing",
            "slot_extraction",
        ],
        "excluded_scope": [
            "clarification",
            "support_abort",
        ],
        "reasoning": (
            "Die lokale Modellvorbereitung konzentriert sich nur auf strukturierte Entscheidungen. "
            "Antworttexte fuer Clarification und Support bleiben deterministisch, damit die Laufzeit "
            "auch ohne Modell stabil bleibt."
        ),
    }
    (package_dir / "router_package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_dir / "router_qlora_config.json").write_text(
        json.dumps(_qlora_config(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(
        "\n".join(
            [
                "# Local Router Package",
                "",
                "Dieses Paket bereitet ein lokales Fine-Tuning nur fuer den strukturierten Semantic-Teil vor.",
                "",
                "Enthalten:",
                "- `router_sft_train.jsonl`",
                "- `router_sft_eval.jsonl`",
                "- `routing_train_only.jsonl`",
                "- `routing_eval_only.jsonl`",
                "- `slot_train_only.jsonl`",
                "- `slot_eval_only.jsonl`",
                "- `router_package_manifest.json`",
                "- `router_qlora_config.json`",
                "",
                "Ziel:",
                "- OpenAI-/FT-Abhaengigkeit fuer `routing` und `slot_extraction` spaeter abloesen",
                "- `clarification` und `support_abort` bewusst weiter deterministisch im Code halten",
                f"- Datensatzbasis: `{dataset_version}`",
                "",
                "Format:",
                "- jedes Beispiel enthaelt `messages` im Chat-Format",
                "- Assistant gibt immer nur JSON zurueck",
                f"- Routing-Modus: `{routing_mode}`",
                "",
                "Naechster Schritt:",
                "- dieses Paket gegen ein lokales 7B-Instruct-Basismodell mit QLoRA trainieren",
                "- danach nur den Router-Teil der Runtime umhaengen, nicht die kompletten Hypnose-Texte",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local router package from a versioned gold split.")
    parser.add_argument("--dataset-version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--routing-mode", choices=list(ROUTING_MODES), default="full")
    args = parser.parse_args()
    manifest = write_local_router_package(
        dataset_version=args.dataset_version,
        routing_mode=args.routing_mode,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
