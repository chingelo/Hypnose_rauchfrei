from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from jsonschema import ValidationError, validate

from build_finetune_splits import DATA_DIR, SCHEMA_DIR
from session_sandbox_orchestrator import SemanticNodeSpec, get_node_spec


DATASET_FILES_BY_VERSION = {
    "v1": {
        "routing": "routing_eval.jsonl",
        "slot_extraction": "slot_extraction_eval.jsonl",
        "clarification": "clarification_eval.jsonl",
        "support_abort": "support_abort_eval.jsonl",
    },
    "v2": {
        "routing": "routing_eval_v2.jsonl",
        "slot_extraction": "slot_extraction_eval_v2.jsonl",
    },
    "v3": {
        "routing": "routing_eval_v3.jsonl",
        "slot_extraction": "slot_extraction_eval_v3.jsonl",
    },
}

SCHEMA_FILES = {
    "routing": "routing_output.schema.json",
    "slot_extraction": "slot_extraction_output.schema.json",
    "clarification": "clarification_output.schema.json",
    "support_abort": "support_abort_output.schema.json",
}


@dataclass(frozen=True)
class ExampleResult:
    dataset: str
    example_id: str
    raw_text: str
    parsed_output: dict[str, Any] | None
    schema_valid: bool
    exact_match: bool
    text_similarity: float | None
    error: str | None


@dataclass(frozen=True)
class DatasetMetrics:
    dataset: str
    total: int
    json_parse_success: int
    schema_valid: int
    exact_match: int
    average_text_similarity: float | None


class CandidateProvider(Protocol):
    def infer(self, dataset: str, prompt: str, expected_output: dict[str, Any]) -> str:
        ...


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"could not parse JSON from model response: {text}")


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _routing_schema_for_example(row: dict[str, Any], *, routing_mode: str) -> dict[str, Any]:
    if routing_mode == "full":
        return _load_json(SCHEMA_DIR / SCHEMA_FILES["routing"])
    spec = get_node_spec(str(row["input"]["node_id"]))
    if not isinstance(spec, SemanticNodeSpec):
        raise TypeError(row["input"]["node_id"])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RoutingIntentOutput",
        "type": "object",
        "additionalProperties": False,
        "required": ["intent"],
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(spec.routing_rules.keys()),
            }
        },
    }


def _build_prompt(
    dataset: str,
    row: dict[str, Any],
    schema: dict[str, Any],
    *,
    routing_mode: str = "full",
) -> str:
    intro = {
        "routing": (
            (
                "Du bist ein strukturiertes Routing-Modell fuer eine deterministische Hypnose-Runtime. "
                "Klassifiziere die Nutzerantwort und gib nur das JSON-Objekt zurueck."
            )
            if routing_mode == "full"
            else (
                "Du bist ein strukturiertes Intent-Klassifikationsmodell fuer eine deterministische Hypnose-Runtime. "
                "Klassifiziere die Nutzerantwort nur auf den kanonischen Intent des aktuellen Knotens und gib nur das JSON-Objekt zurueck."
            )
        ),
        "slot_extraction": (
            "Du bist ein strukturiertes Extraktionsmodell fuer eine deterministische Hypnose-Runtime. "
            "Extrahiere nur die benoetigten Slots und gib nur das JSON-Objekt zurueck."
        ),
        "clarification": (
            "Du bist ein strukturiertes Antwortmodell fuer kurze therapeutische Klaerungen. "
            "Formuliere genau eine passende Rueckmeldung und gib nur das JSON-Objekt zurueck."
        ),
        "support_abort": (
            "Du bist ein strukturiertes Antwortmodell fuer Support- und Abbruchtexte. "
            "Formuliere genau eine passende Rueckmeldung und gib nur das JSON-Objekt zurueck."
        ),
    }[dataset]

    payload: dict[str, Any] = {
        "task": dataset,
        "example_id": row["id"],
        "input": row["input"],
        "required_schema": schema,
        "rules": [
            "Nur ein JSON-Objekt ausgeben.",
            "Keine Markdown-Blocks, keine Erklaerungen, kein Zusatztext.",
            "Keine Felder ausserhalb des Schemas verwenden.",
            "Nutze nur die Informationen aus der Eingabe.",
        ],
    }
    if dataset == "routing" and routing_mode == "intent_only":
        payload["task"] = "routing_intent"
        payload["rules"].append("Nur das Feld `intent` ausgeben, nicht `action` oder `next_node`.")
    return intro + "\n\n" + json.dumps(payload, ensure_ascii=True, indent=2)


def _schema_for_dataset(dataset: str) -> dict[str, Any]:
    return _load_json(SCHEMA_DIR / SCHEMA_FILES[dataset])


def _dataset_dir(version: str) -> Path:
    if version == "v1":
        return DATA_DIR
    if version in {"v2", "v3"}:
        return DATA_DIR / version
    raise ValueError(version)


def _examples_for_dataset(dataset: str, *, version: str, limit: int | None = None) -> list[dict[str, Any]]:
    dataset_files = DATASET_FILES_BY_VERSION[version]
    rows = _load_jsonl(_dataset_dir(version) / dataset_files[dataset])
    return rows[:limit] if limit is not None else rows


class ReferenceProvider:
    def infer(self, dataset: str, prompt: str, expected_output: dict[str, Any]) -> str:
        del dataset, prompt
        return json.dumps(expected_output, ensure_ascii=True)


class OllamaProvider:
    def __init__(self, model: str, endpoint: str, timeout_seconds: float) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def infer(self, dataset: str, prompt: str, expected_output: dict[str, Any]) -> str:
        del dataset, expected_output
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            url=f"{self.endpoint}/api/generate",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        if "response" not in payload:
            raise RuntimeError(f"Unexpected Ollama payload: {payload}")
        return str(payload["response"])


class LocalAdapterProvider:
    def __init__(
        self,
        *,
        adapter_dir: Path,
        base_model: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.adapter_dir = adapter_dir
        adapter_config = _load_json(adapter_dir / "adapter_config.json")
        self.base_model = base_model or str(adapter_config["base_model_name_or_path"])
        self.max_new_tokens = max_new_tokens

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            local_files_only=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True)
        self.model.eval()

    def infer(self, dataset: str, prompt: str, expected_output: dict[str, Any]) -> str:
        del dataset, expected_output
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = tokenized["input_ids"].shape[1]
        completion_ids = generated[0][prompt_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _text_similarity(expected: str, actual: str) -> float:
    return SequenceMatcher(None, _normalize_text(expected), _normalize_text(actual)).ratio()


def _compare_output(
    dataset: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    row: dict[str, Any] | None = None,
    routing_mode: str = "full",
) -> tuple[bool, float | None]:
    if dataset in {"routing", "slot_extraction"}:
        if dataset == "routing" and routing_mode == "intent_only":
            if row is None:
                raise ValueError("row is required for routing intent comparison")
            spec = get_node_spec(str(row["input"]["node_id"]))
            if not isinstance(spec, SemanticNodeSpec):
                raise TypeError(row["input"]["node_id"])
            intent = str(actual.get("intent", "")).strip()
            route = spec.routing_rules.get(intent)
            if not route:
                return False, None
            normalized = {
                "intent": intent,
                "action": route["action"],
                "next_node": route["next_node"],
            }
            return expected == normalized, None
        return expected == actual, None
    expected_text = str(expected.get("reply_text", ""))
    actual_text = str(actual.get("reply_text", ""))
    similarity = _text_similarity(expected_text, actual_text)
    return _normalize_text(expected_text) == _normalize_text(actual_text), similarity


def _evaluate_dataset(
    dataset: str,
    provider: CandidateProvider,
    *,
    version: str,
    limit: int | None = None,
    routing_mode: str = "full",
) -> tuple[DatasetMetrics, list[ExampleResult]]:
    rows = _examples_for_dataset(dataset, version=version, limit=limit)
    results: list[ExampleResult] = []

    for row in rows:
        schema = (
            _routing_schema_for_example(row, routing_mode=routing_mode)
            if dataset == "routing"
            else _schema_for_dataset(dataset)
        )
        prompt = _build_prompt(dataset, row, schema, routing_mode=routing_mode)
        raw_text = ""
        parsed_output: dict[str, Any] | None = None
        schema_valid = False
        exact_match = False
        similarity: float | None = None
        error: str | None = None

        try:
            raw_text = provider.infer(dataset, prompt, row["output"])
            parsed_output = parse_json_object(raw_text)
            validate(instance=parsed_output, schema=schema)
            schema_valid = True
            exact_match, similarity = _compare_output(
                dataset,
                row["output"],
                parsed_output,
                row=row,
                routing_mode=routing_mode,
            )
        except (ValueError, ValidationError, RuntimeError, json.JSONDecodeError) as exc:
            error = str(exc)

        results.append(
            ExampleResult(
                dataset=dataset,
                example_id=row["id"],
                raw_text=raw_text,
                parsed_output=parsed_output,
                schema_valid=schema_valid,
                exact_match=exact_match,
                text_similarity=similarity,
                error=error,
            )
        )

    similarity_values = [result.text_similarity for result in results if result.text_similarity is not None]
    metrics = DatasetMetrics(
        dataset=dataset,
        total=len(results),
        json_parse_success=sum(result.parsed_output is not None for result in results),
        schema_valid=sum(result.schema_valid for result in results),
        exact_match=sum(result.exact_match for result in results),
        average_text_similarity=(
            round(sum(similarity_values) / len(similarity_values), 4) if similarity_values else None
        ),
    )
    return metrics, results


def _build_provider(args: argparse.Namespace) -> CandidateProvider:
    if args.provider == "reference":
        return ReferenceProvider()
    if args.provider == "ollama":
        if not args.model:
            raise SystemExit("--model ist fuer --provider ollama erforderlich.")
        return OllamaProvider(
            model=args.model,
            endpoint=args.ollama_endpoint,
            timeout_seconds=args.timeout_seconds,
        )
    if args.provider == "local_adapter":
        if not args.adapter_dir:
            raise SystemExit("--adapter-dir ist fuer --provider local_adapter erforderlich.")
        return LocalAdapterProvider(
            adapter_dir=Path(args.adapter_dir),
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
        )
    raise SystemExit(f"Unsupported provider: {args.provider}")


def _report_path(args: argparse.Namespace) -> Path:
    if args.report_path:
        return Path(args.report_path)
    if args.provider == "ollama":
        suffix = f"ollama_{args.model.replace(':', '_')}"
    elif args.provider == "local_adapter":
        suffix = f"local_adapter_{Path(args.adapter_dir).name}"
    else:
        suffix = args.provider
    if args.routing_mode == "intent_only":
        suffix += "_intent_only"
    return _dataset_dir(args.dataset_version) / f"eval_report_{args.dataset_version}_{suffix}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a structured finetune candidate against local eval splits.")
    parser.add_argument(
        "--provider",
        choices=["reference", "ollama", "local_adapter"],
        default="reference",
        help="reference = erwartete Outputs direkt zur Harness-Validierung, ollama = lokales Modell ueber Ollama, local_adapter = lokaler HF/PEFT-Adapter.",
    )
    parser.add_argument("--model", help="Lokales Ollama-Modell, z. B. mistral:7b-instruct")
    parser.add_argument("--adapter-dir", help="Pfad zu einem lokal trainierten PEFT-Adapter.")
    parser.add_argument("--base-model", help="Optionaler Override fuer das Base-Modell eines lokalen Adapters.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted({key for mapping in DATASET_FILES_BY_VERSION.values() for key in mapping}),
        default=list(DATASET_FILES_BY_VERSION["v1"].keys()),
        help="Welche Eval-Sets ausgewertet werden sollen.",
    )
    parser.add_argument(
        "--dataset-version",
        choices=sorted(DATASET_FILES_BY_VERSION),
        default="v1",
        help="Welcher Eval-Stand genutzt werden soll.",
    )
    parser.add_argument("--limit", type=int, help="Optionale Begrenzung pro Dataset.")
    parser.add_argument("--report-path", help="Optionaler Zielpfad fuer den JSON-Report.")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434", help="Lokaler Ollama-Endpunkt.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Timeout pro Modellaufruf.")
    parser.add_argument("--max-failures", type=int, default=10, help="Maximale Anzahl Beispiel-Fehler im Report.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximale Generationslaenge fuer local_adapter.")
    parser.add_argument(
        "--routing-mode",
        choices=["full", "intent_only"],
        default="full",
        help="Volles Routing-JSON oder Intent-only-Klassifikation fuer Routing.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    provider = _build_provider(args)
    report_file = _report_path(args)
    available = DATASET_FILES_BY_VERSION[args.dataset_version]
    unsupported = [dataset for dataset in args.datasets if dataset not in available]
    if unsupported:
        raise SystemExit(
            f"Datensaetze fuer {args.dataset_version} nicht verfuegbar: {', '.join(unsupported)}"
        )

    metrics: list[DatasetMetrics] = []
    failures: list[dict[str, Any]] = []

    for dataset in args.datasets:
        dataset_metrics, results = _evaluate_dataset(
            dataset,
            provider,
            version=args.dataset_version,
            limit=args.limit,
            routing_mode=args.routing_mode,
        )
        metrics.append(dataset_metrics)
        for result in results:
            if result.exact_match and result.schema_valid:
                continue
            failures.append(asdict(result))

    report = {
        "provider": args.provider,
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "base_model": args.base_model,
        "dataset_version": args.dataset_version,
        "routing_mode": args.routing_mode,
        "datasets": args.datasets,
        "limit": args.limit,
        "metrics": [asdict(metric) for metric in metrics],
        "failure_count": len(failures),
        "failures": failures[: args.max_failures],
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Report written to: {report_file}")
    for metric in metrics:
        line = (
            f"{metric.dataset}: total={metric.total} parse={metric.json_parse_success}/{metric.total} "
            f"schema={metric.schema_valid}/{metric.total} exact={metric.exact_match}/{metric.total}"
        )
        if metric.average_text_similarity is not None:
            line += f" avg_text_similarity={metric.average_text_similarity}"
        print(line)
    print(f"failure_count={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
