from __future__ import annotations

import argparse
from pathlib import Path

from phase4_node_engine import SPEC_DIR, advance_node, load_node_spec


def spec_exists(node_id: str) -> bool:
    return (SPEC_DIR / f"{node_id}.json").exists()


def resolve_control_next_node(current_node_id: str, next_node_id: str) -> tuple[str, bool]:
    if next_node_id in {"clarify_same_node", "repeat_same_question"}:
        return current_node_id, True
    return next_node_id, False


def print_block(title: str, text: str) -> None:
    print(f"\n[{title}]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-4 decision-node prototype runner")
    parser.add_argument("--node", default="pleasantness", help="Start node id")
    args = parser.parse_args()

    node_id = args.node
    if not spec_exists(node_id):
        available = ", ".join(sorted(path.stem for path in Path(SPEC_DIR).glob("*.json")))
        raise SystemExit(f"Unknown start node '{node_id}'. Available nodes: {available}")

    print("Phase-4 Node Prototype")
    print("Type 'exit' to quit.")

    while True:
        spec = load_node_spec(node_id)
        print_block(f"NODE {node_id}", spec["question_text"])
        user_text = input("> ").strip()
        if user_text.lower() in {"exit", "quit", "abbrechen"}:
            print("Prototype stopped.")
            return

        decision = advance_node(node_id, user_text)
        print_block("PARSE", f"intent={decision.parse_result.intent} | confidence={decision.parse_result.confidence} | reason={decision.parse_result.reason}")
        print_block("REPLY", decision.reply_text)
        print_block("NEXT", decision.next_node)

        resolved_next_node, is_control_flow = resolve_control_next_node(node_id, decision.next_node)
        if is_control_flow:
            node_id = resolved_next_node
            continue

        if decision.next_node == "abort_confirmation":
            print("Prototype stopped after abort confirmation.")
            return

        if not spec_exists(resolved_next_node):
            print(f"No local spec for next node '{decision.next_node}'. Prototype stops here.")
            return

        node_id = resolved_next_node


if __name__ == "__main__":
    main()
