"""End-to-end assistant in the terminal — no Streamlit needed.

Interactive:
    python scripts/demo_cli.py

Scripted run that exercises every acceptance criterion in one session
(useful for generating docs/sample_questions.md and for the recording):
    python scripts/demo_cli.py --scripted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.memory import ConversationMemory  # noqa: E402
from agent.pipeline import answer_question  # noqa: E402
from config import DESTINATION, OLLAMA_MODEL  # noqa: E402

SCRIPT = [
    # pure RAG
    "What are the must-visit attractions in Singapore?",
    "Which neighbourhoods are best for cultural experiences?",
    # pure MCP — weather
    "What is the forecast for the next three days?",
    # pure MCP — currency
    "Convert INR 50,000 to SGD.",
    # combined (the required scenario)
    "Create a three-day Singapore itinerary for next week and adjust it to the weather forecast.",
    # multi-turn context: no destination, no budget restated
    "I'm travelling with two kids aged 6 and 9, and my budget is INR 60,000.",
    "Given that, revise the itinerary and show the budget in the local currency.",
    # knowledge gap
    "What are the best ski resorts in Singapore?",
]


def print_trace(trace) -> None:
    plan = trace.plan
    print(f"    planner: {plan.get('planner')} | {plan.get('reasoning')}")
    print(f"    kb query: {plan.get('kb_query') or '-'}")
    if trace.retrieval:
        kept = trace.retrieval.get("kept") or []
        if kept:
            print(
                "    retrieved: "
                + "; ".join(f"{h['label']} {h['source_title']} ({h['score']})" for h in kept)
            )
        else:
            print("    retrieved: nothing above threshold -> refusal path")
    for call in trace.tool_calls:
        print(f"    tool: {call['name']}({call['arguments']}) -> {call['status']}")
    print(f"    timings: {trace.timings}")


def ask(question: str, memory: ConversationMemory) -> None:
    print("\n" + "=" * 78)
    print(f"USER: {question}")
    print("-" * 78)
    answer = answer_question(question, memory)
    print_trace(answer.trace)
    if answer.preference_updates:
        print(f"    preferences updated: {answer.preference_updates}")
    print("-" * 78)
    print(answer.text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args()

    memory = ConversationMemory()
    print(f"{DESTINATION} Travel Assistant — model: {OLLAMA_MODEL}")

    if args.scripted:
        for question in SCRIPT:
            ask(question, memory)
        print("\nFinal remembered preferences:", memory.preferences_text())
        return 0

    print("Type a question, or 'quit' to exit, 'reset' to clear memory.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return 0
        if question.lower() == "reset":
            memory.reset()
            print("memory cleared.\n")
            continue
        ask(question, memory)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
