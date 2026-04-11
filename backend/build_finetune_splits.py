from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "finetune_data"
SCHEMA_DIR = DATA_DIR / "schemas"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _stable_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(row: dict[str, Any]) -> str:
        return hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()

    return sorted(rows, key=_key)


def _split_groups(
    rows: list[dict[str, Any]],
    *,
    group_key: Callable[[dict[str, Any]], str],
    eval_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for key in sorted(grouped):
        ordered = _stable_order(grouped[key])
        size = len(ordered)
        if size <= 1:
            train.extend(ordered)
            continue
        if size <= 3:
            eval_count = 1
        else:
            eval_count = max(1, round(size * eval_ratio))
            if eval_count >= size:
                eval_count = size - 1
        eval_rows.extend(ordered[:eval_count])
        train.extend(ordered[eval_count:])

    return train, eval_rows


def _routing_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RoutingModelOutput",
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "action", "next_node"],
        "properties": {
            "intent": {"type": "string", "minLength": 1},
            "action": {
                "type": "string",
                "enum": ["transition", "clarify", "repeat", "answer_question", "support", "abort"],
            },
            "next_node": {"type": "string", "minLength": 1},
        },
    }


def _slot_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SlotExtractionOutput",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "focus_kind": {"type": "string", "enum": ["person", "group", "other", "unknown"]},
            "named_person": {"type": ["string", "null"]},
            "display_trigger_focus_ref": {"type": "string"},
        },
        "minProperties": 1,
    }


def _reply_schema(title: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "required": ["reply_text"],
        "properties": {
            "reply_text": {"type": "string", "minLength": 1},
        },
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    return {
        "routing_output.schema.json": _routing_schema(),
        "slot_extraction_output.schema.json": _slot_schema(),
        "clarification_output.schema.json": _reply_schema("ClarificationReplyOutput"),
        "support_abort_output.schema.json": _reply_schema("SupportAbortReplyOutput"),
    }


def build_splits() -> dict[str, list[dict[str, Any]]]:
    routing = _load_jsonl(DATA_DIR / "routing_gold.jsonl")
    slot = _load_jsonl(DATA_DIR / "slot_extraction_gold.jsonl")
    clarification = _load_jsonl(DATA_DIR / "clarification_gold.jsonl")
    support = _load_jsonl(DATA_DIR / "support_abort_gold.jsonl")

    routing_train, routing_eval = _split_groups(
        routing,
        group_key=lambda row: str(row["input"]["node_id"]),
        eval_ratio=0.2,
    )
    slot_train, slot_eval = _split_groups(
        slot,
        group_key=lambda row: str(row["task"]),
        eval_ratio=0.2,
    )
    clarification_train, clarification_eval = _split_groups(
        clarification,
        group_key=lambda row: str(row["task"]),
        eval_ratio=0.2,
    )
    support_train, support_eval = _split_groups(
        support,
        group_key=lambda row: str(row["task"]),
        eval_ratio=0.2,
    )

    return {
        "routing_train.jsonl": routing_train,
        "routing_eval.jsonl": routing_eval,
        "slot_extraction_train.jsonl": slot_train,
        "slot_extraction_eval.jsonl": slot_eval,
        "clarification_train.jsonl": clarification_train,
        "clarification_eval.jsonl": clarification_eval,
        "support_abort_train.jsonl": support_train,
        "support_abort_eval.jsonl": support_eval,
    }


def write_schemas() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, schema in build_schemas().items():
        (SCHEMA_DIR / name).write_text(
            json.dumps(schema, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )


def write_manifest(splits: dict[str, list[dict[str, Any]]]) -> None:
    manifest = {
        "version": 1,
        "generated_from": "gold_datasets_v1",
        "files": {name: len(rows) for name, rows in splits.items()},
    }
    (DATA_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    splits = build_splits()
    for name, rows in splits.items():
        _write_jsonl(DATA_DIR / name, rows)
        print(f"{name}: {len(rows)}")
    write_schemas()
    write_manifest(splits)


if __name__ == "__main__":
    main()
