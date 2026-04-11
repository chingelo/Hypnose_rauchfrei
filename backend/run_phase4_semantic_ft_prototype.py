from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from live_api_guard import DEFAULT_APPROVAL_FILE, LiveApiCallBudget, build_live_api_budget
from openai_semantic_backend import OpenAISemanticBackend, resolve_ft_backend, resolve_openai_router_backend
from phase4_semantic_prompt_prototype import (
    SemanticNodeSpec,
    ScriptNodeSpec,
    available_node_ids,
    build_request,
    get_node_spec,
    get_semantic_node_spec,
    repair_semantic_payload,
    script_reply_for_decision,
    validate_semantic_decision,
)


ENV_PATHS = (
    Path(r"C:\Projekte\test_app\backend\.env"),
)


@dataclass(frozen=True)
class BatchCase:
    node_id: str
    customer_message: str
    expected_next_node: str


SCENARIOS: dict[str, list[BatchCase]] = {
    "hell_feel_branch_variants": [
        BatchCase("hell_feel_branch", "sehr angenehm", "hell_hypnose_loch_intro"),
        BatchCase("hell_feel_branch", "das fuehlt sich angenehm an", "hell_hypnose_loch_intro"),
        BatchCase("hell_feel_branch", "warm und angenehm", "hell_hypnose_loch_intro"),
        BatchCase("hell_feel_branch", "eher unangenehm", "hell_regulation_choice"),
        BatchCase("hell_feel_branch", "das drueckt mich eher", "hell_regulation_choice"),
        BatchCase("hell_feel_branch", "ich weiss nicht", "hell_feel_branch"),
        BatchCase("hell_feel_branch", "kannst du das kurz erklaeren", "hell_feel_branch"),
        BatchCase("hell_feel_branch", "das ist mir gerade etwas zu viel", "hell_feel_branch"),
    ],
    "pleasant_flow_to_resolved": [
        BatchCase("hell_feel_branch", "das fuehlt sich sehr angenehm an", "hell_hypnose_loch_intro"),
        BatchCase("hell_hypnose_wait", "es loest sich noch auf", "hell_hypnose_wait"),
        BatchCase("hell_hypnose_wait", "jetzt ist es aufgeloest", "hell_post_resolved_terminal"),
    ],
    "dark_known_branch_variants": [
        BatchCase("dark_known_branch", "das kenne ich schon von frueher", "dark_backtrace_terminal"),
        BatchCase("dark_known_branch", "das war schon frueher immer wieder da", "dark_backtrace_terminal"),
        BatchCase("dark_known_branch", "hier ist es zum ersten Mal", "dark_origin_terminal"),
        BatchCase("dark_known_branch", "ich glaube das ist der erste Ursprung", "dark_origin_terminal"),
        BatchCase("dark_known_branch", "ich weiss es nicht", "dark_known_branch"),
        BatchCase("dark_known_branch", "kannst du das kurz erklaeren", "dark_known_branch"),
        BatchCase("dark_known_branch", "das ist mir gerade zu viel", "dark_known_branch"),
    ],
    "pleasant_flow_to_dark_origin": [
        BatchCase("hell_feel_branch", "warm und angenehm", "hell_hypnose_loch_intro"),
        BatchCase("hell_hypnose_wait", "jetzt ist es aufgeloest", "hell_post_resolved_terminal"),
        BatchCase("dark_known_branch", "hier ist es zum ersten Mal", "dark_origin_terminal"),
    ],
    "unpleasant_flow": [
        BatchCase("hell_feel_branch", "das drueckt mich eher", "hell_regulation_choice"),
        BatchCase("hell_regulation_choice", "mehr Abstand", "hell_regulation_check"),
        BatchCase("hell_regulation_check", "deutlich ruhiger", "dark_known_branch"),
    ],
}


def load_env() -> None:
    for path in ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=False)


def resolve_openai_client() -> tuple[OpenAI, str]:
    backend = resolve_ft_backend()
    return backend.client, backend.model


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


def call_semantic_node(
    openai_backend: OpenAISemanticBackend,
    node_id: str,
    customer_message: str,
    *,
    clarify_attempt: int = 0,
    session_context: str = "",
    live_api_budget: LiveApiCallBudget | None = None,
) -> tuple[dict[str, Any], Any]:
    spec = get_semantic_node_spec(node_id)
    payload = build_request(
        node_id,
        customer_message,
        clarify_attempt=clarify_attempt,
        session_context=session_context,
    )
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    effective_system_prompt = openai_backend.compose_semantic_system_prompt(spec.system_prompt)

    try:
        parsed = openai_backend.infer_semantic_json(
            system_prompt=effective_system_prompt,
            user_payload=payload,
            live_api_budget=live_api_budget,
        )
    except Exception:
        client = openai_backend.client
        model = openai_backend.model
        if live_api_budget is not None:
            live_api_budget.consume(f"{node_id}:chat_fallback")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **openai_backend.chat_completion_options(),
        )
        content = completion.choices[0].message.content or ""
        parsed = parse_json_object(content)
    parsed = repair_semantic_payload(node_id, parsed)
    decision = validate_semantic_decision(node_id, parsed)
    return parsed, decision


def resolve_openai_backend(provider: str) -> OpenAISemanticBackend:
    if provider == "ft":
        return resolve_ft_backend()
    if provider == "openai-router":
        return resolve_openai_router_backend()
    raise RuntimeError(f"Unsupported provider: {provider}")


def render_script_node(node_id: str) -> tuple[str, str | None]:
    spec = get_node_spec(node_id)
    if not isinstance(spec, ScriptNodeSpec):
        raise ValueError(f"node '{node_id}' is not a script node")
    return spec.script_text, spec.next_node


def maybe_render_entry_script(node_id: str) -> str:
    spec = get_node_spec(node_id)
    if isinstance(spec, SemanticNodeSpec):
        return spec.entry_script.strip()
    return ""


def estimate_batch_live_api_calls(scenario_name: str) -> int:
    return len(SCENARIOS.get(scenario_name) or [])


def run_batch(
    scenario_name: str,
    *,
    debug_model: bool = False,
    live_api_budget: LiveApiCallBudget | None = None,
    provider: str = "ft",
) -> int:
    openai_backend = resolve_openai_backend(provider)
    scenario = SCENARIOS.get(scenario_name) or []
    if not scenario:
        available = ", ".join(sorted(SCENARIOS))
        print(f"Unknown scenario '{scenario_name}'. Available: {available}")
        return 1

    session_context = ""
    failures = 0
    for case in scenario:
        entry_script = maybe_render_entry_script(case.node_id)
        if entry_script:
            print(f"ENTRY {case.node_id}: {entry_script}")
        parsed, decision = call_semantic_node(
            openai_backend,
            case.node_id,
            case.customer_message,
            session_context=session_context,
            live_api_budget=live_api_budget,
        )
        script_reply = script_reply_for_decision(case.node_id, decision)
        success = decision.next_node == case.expected_next_node
        if not success:
            failures += 1
        print(f"NODE: {case.node_id}")
        print(f"INPUT: {case.customer_message}")
        if debug_model:
            print(f"MODEL: {json.dumps(parsed, ensure_ascii=False)}")
        print(f"DECISION: intent={decision.intent} action={decision.action} next={decision.next_node} confidence={decision.confidence}")
        if script_reply:
            print(f"SCRIPT_REPLY: {script_reply}")
        if decision.next_node in available_node_ids():
            next_spec = get_node_spec(decision.next_node)
            if isinstance(next_spec, ScriptNodeSpec):
                print(f"NEXT_SCRIPT: {next_spec.script_text}")
            elif next_spec.question_text:
                print(f"NEXT_QUESTION: {next_spec.question_text}")
        print(f"EXPECTED_NEXT: {case.expected_next_node}")
        print(f"RESULT: {'PASS' if success else 'FAIL'}")
        print('-' * 100)
        session_context = script_reply or session_context
    return 1 if failures else 0


def run_interactive(start_node: str, *, debug_model: bool = False, provider: str = "ft") -> int:
    openai_backend = resolve_openai_backend(provider)
    node_id = start_node
    session_context = ""
    clarify_attempts: dict[str, int] = {}
    last_node_id: str | None = None

    print("Semantic FT Prototype")
    print("Type 'exit' to quit.")

    while True:
        current_spec = get_node_spec(node_id)
        if isinstance(current_spec, ScriptNodeSpec):
            print(f"\n[SCRIPT]\n{current_spec.script_text}\n")
            if current_spec.next_node is None:
                print("Prototype finished.")
                return 0
            last_node_id = node_id
            node_id = current_spec.next_node
            continue

        if node_id != last_node_id and current_spec.entry_script:
            print(f"\n[SCRIPT]\n{current_spec.entry_script}\n")

        print(f"\n[NODE {node_id}]\n{current_spec.question_text}\n")
        user_text = input("> ").strip()
        if user_text.lower() in {"exit", "quit", "abbrechen"}:
            print("Prototype stopped.")
            return 0

        clarify_attempt = clarify_attempts.get(node_id, 0)
        parsed, decision = call_semantic_node(
            openai_backend,
            node_id,
            user_text,
            clarify_attempt=clarify_attempt,
            session_context=session_context,
        )
        if debug_model:
            print(f"\n[MODEL]\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n")
        print(
            f"[DECISION]\nintent={decision.intent} action={decision.action} next={decision.next_node} confidence={decision.confidence}\n"
        )

        script_reply = script_reply_for_decision(node_id, decision)
        if script_reply:
            print(f"[SCRIPT]\n{script_reply}\n")
            session_context = script_reply

        if decision.next_node == node_id:
            clarify_attempts[node_id] = clarify_attempt + 1
        else:
            clarify_attempts[node_id] = 0

        last_node_id = node_id
        node_id = decision.next_node


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic FT prototype runner for Phase 4")
    parser.add_argument("--node", default="hell_feel_branch", help="Start node id")
    parser.add_argument("--batch", action="store_true", help="Run predefined free-text scenario")
    parser.add_argument("--scenario", default="hell_feel_branch_variants", help="Scenario name for --batch")
    parser.add_argument("--debug-model", action="store_true", help="Show raw model JSON")
    parser.add_argument("--provider", choices=["ft", "openai-router"], default="ft", help="Altes FT oder aktueller OpenAI-Router.")
    parser.add_argument("--live-api", action="store_true", help="Erlaubt fuer --batch den echten OpenAI-Lauf.")
    parser.add_argument("--max-api-calls", type=int, help="Verpflichtendes hartes Limit fuer Live-OpenAI-Calls.")
    parser.add_argument(
        "--approval-file",
        default=str(DEFAULT_APPROVAL_FILE),
        help="Pfad zur Repo-Freigabedatei fuer Live-OpenAI.",
    )
    args = parser.parse_args()

    if args.node not in available_node_ids():
        available = ", ".join(sorted(available_node_ids()))
        raise SystemExit(f"Unknown start node '{args.node}'. Available: {available}")

    if args.batch:
        estimated_calls = estimate_batch_live_api_calls(args.scenario)
        if not args.live_api:
            print("Live-OpenAI ist fuer den FT-Batch standardmaessig blockiert.")
            print(f"Geschaetzte Live-Calls fuer dieses Szenario: {estimated_calls}")
            print(
                "Zum bewussten Live-Lauf braucht es: OPENAI_LIVE_API_ALLOWED=1, "
                "eine gueltige backend/live_api_approval.json und --live-api --max-api-calls <n>."
            )
            return 0
        live_api_budget = build_live_api_budget(
            "run_phase4_semantic_ft_prototype.py",
            estimated_calls=estimated_calls,
            requested_max_calls=args.max_api_calls,
            approval_file=args.approval_file,
        )
        print(live_api_budget.summary())
        return run_batch(args.scenario, debug_model=args.debug_model, live_api_budget=live_api_budget, provider=args.provider)
    return run_interactive(args.node, debug_model=args.debug_model, provider=args.provider)


if __name__ == "__main__":
    raise SystemExit(main())
