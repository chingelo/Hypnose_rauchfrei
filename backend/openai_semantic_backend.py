from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from live_api_guard import LiveApiCallBudget


ENV_PATHS = (
    Path(r"C:\Projekte\test_app\backend\.env"),
)


def _load_env_with_bom_fallback(path: Path) -> None:
    load_dotenv(path, override=False)
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_key"):
        return

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value


class SemanticDecisionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    action: str
    next_node: str
    confidence: float | None = None


class TextReplyEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str


def sanitize_model_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            normalized = sanitize_model_json(child)
            if normalized is not None:
                sanitized[key] = normalized
        return sanitized
    if isinstance(value, list):
        return [sanitize_model_json(child) for child in value]
    return value


def sanitize_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = sanitize_model_json(payload)
    confidence = normalized.get("confidence")
    if confidence is not None:
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            normalized.pop("confidence", None)
        else:
            if 0.0 <= confidence_value <= 1.0:
                normalized["confidence"] = confidence_value
            else:
                normalized.pop("confidence", None)
    return normalized


@dataclass(frozen=True)
class OpenAISemanticBackend:
    client: OpenAI
    model: str
    api_mode: str
    profile: str = "generic"
    reasoning_effort: str | None = None
    verbosity: str | None = None

    def compose_semantic_system_prompt(
        self,
        base_prompt: str,
        *,
        runtime_question: str | None = None,
    ) -> str:
        if self.profile != "router":
            return base_prompt

        router_prefix = (
            "Du bist kein freier Assistent und kein freier Therapeut, sondern ein streng deterministischer "
            "semantischer Router fuer eine Hypnose-Runtime.\n"
            "Entscheide nur innerhalb des aktuellen Knotens.\n"
            "Der knotspezifische Prompt unten ist Ground Truth. Wenn allgemeine Hypnoseannahmen und der Knotenprompt "
            "abweichen, gilt immer der Knotenprompt.\n"
            "Erfinde niemals neue Labels, neue Knoten, neue Phasen oder neue Logik.\n"
            "Wenn die Nutzerantwort die aktuelle Frage nicht klar trifft, bleibe bei den vorgesehenen Safe-Intents "
            "wie `unclear`, `question`, `support_needed`, `repeat` oder `abort`, sofern sie in diesem Knoten erlaubt sind.\n"
            "Rate niemals nur aus Kontext oder Slots einen Inhalts-Branch, wenn die aktuelle Antwort ihn nicht selbst traegt.\n"
            "Gib nur das strukturierte Ergebnis zurueck. Kein Fliesstext, keine Erklaerung, keine therapeutische Antwort."
        )
        question_block = ""
        if runtime_question:
            question_block = f"\n\nKonkrete Laufzeitfrage fuer diesen Durchlauf:\n\"{runtime_question}\""
        return f"{router_prefix}\n\nKnotenspezifische Ground-Truth-Regeln:\n{base_prompt}{question_block}"

    def _consume(self, live_api_budget: LiveApiCallBudget | None, label: str) -> None:
        if live_api_budget is not None:
            live_api_budget.consume(label)

    def _chat_completion_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.profile != "router":
            options["temperature"] = 0
        return options

    def chat_completion_options(self) -> dict[str, Any]:
        return dict(self._chat_completion_options())

    def infer_semantic_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        live_api_budget: LiveApiCallBudget | None = None,
    ) -> dict[str, Any]:
        if self.api_mode == "responses":
            self._consume(live_api_budget, "semantic:responses")
            options: dict[str, Any] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": json.dumps(user_payload, ensure_ascii=True),
                "text_format": SemanticDecisionEnvelope,
            }
            if self.reasoning_effort:
                options["reasoning"] = {"effort": self.reasoning_effort}
            if self.verbosity:
                options["text"] = {"verbosity": self.verbosity}
            response = self.client.responses.parse(**options)
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Responses API lieferte kein geparstes Structured Output.")
            return sanitize_semantic_payload(parsed.model_dump(exclude_none=True))

        self._consume(live_api_budget, "semantic:chat_completions")
        completion = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            **self._chat_completion_options(),
        )
        content = completion.choices[0].message.content or ""
        return sanitize_semantic_payload(json.loads(content))

    def generate_text_reply(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        live_api_budget: LiveApiCallBudget | None = None,
    ) -> str:
        if self.api_mode == "responses":
            self._consume(live_api_budget, "text:responses")
            options: dict[str, Any] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": json.dumps(user_payload, ensure_ascii=True),
                "text_format": TextReplyEnvelope,
            }
            if self.reasoning_effort:
                options["reasoning"] = {"effort": self.reasoning_effort}
            if self.verbosity:
                options["text"] = {"verbosity": self.verbosity}
            response = self.client.responses.parse(**options)
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Responses API lieferte keinen geparsten Text-Reply.")
            return parsed.reply.strip()

        self._consume(live_api_budget, "text:chat_completions")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            **self._chat_completion_options(),
        )
        return (completion.choices[0].message.content or "").strip()


def load_env() -> None:
    for path in ENV_PATHS:
        if path.exists():
            _load_env_with_bom_fallback(path)


def _resolve_common_openai_client() -> OpenAI:
    load_env()
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_key") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY fehlt. Erwarteter Pfad: backend/.env")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def resolve_ft_backend() -> OpenAISemanticBackend:
    client = _resolve_common_openai_client()
    model = (os.getenv("HYPNOSE_MODEL_ID") or "ft:gpt-3.5-turbo-1106:personal::AzSLcCUs").strip()
    return OpenAISemanticBackend(client=client, model=model, api_mode="chat_completions", profile="ft")


def resolve_openai_router_backend() -> OpenAISemanticBackend:
    client = _resolve_common_openai_client()
    model = (os.getenv("HYPNOSE_ROUTER_MODEL_ID") or "gpt-5-mini").strip()
    api_mode = (os.getenv("HYPNOSE_ROUTER_API_MODE") or "responses").strip().lower()
    if api_mode not in {"responses", "chat_completions"}:
        raise RuntimeError("HYPNOSE_ROUTER_API_MODE muss 'responses' oder 'chat_completions' sein.")
    reasoning_effort = (os.getenv("HYPNOSE_ROUTER_REASONING_EFFORT") or "low").strip().lower() or None
    verbosity = (os.getenv("HYPNOSE_ROUTER_VERBOSITY") or "low").strip().lower() or None
    return OpenAISemanticBackend(
        client=client,
        model=model,
        api_mode=api_mode,
        profile="router",
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )
