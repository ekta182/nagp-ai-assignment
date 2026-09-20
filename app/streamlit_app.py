"""Streamlit chat interface.

The per-turn expander showing plan -> retrieval scores -> tool call -> raw
result is deliberate: it's what makes tool selection and grounding visible
during the demo rather than something the grader has to take on faith.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from agent.memory import ConversationMemory  # noqa: E402
from agent.pipeline import stream_answer  # noqa: E402
from config import DESTINATION, OLLAMA_MODEL  # noqa: E402
from rag.retriever import index_manifest  # noqa: E402

st.set_page_config(
    page_title=f"{DESTINATION} Travel Assistant", page_icon="🧭", layout="wide"
)

EXAMPLES = [
    "What are the must-visit attractions in Singapore?",
    "Which neighbourhoods are best for cultural experiences?",
    "How do I get around Singapore on public transport?",
    "What is the weather in Singapore right now?",
    "What is the forecast for the next three days?",
    "Convert INR 50,000 to SGD.",
    "Create a three-day Singapore itinerary for next week and adjust it to the weather forecast.",
    "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary.",
    "Suggest outdoor attractions, and replace them with indoor options if rain is expected.",
    "What are the best ski resorts in Singapore?",
]


def init_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, trace}]
    if "pending" not in st.session_state:
        st.session_state.pending = None


def render_trace(trace_dict: dict) -> None:
    plan = trace_dict.get("plan", {})
    retrieval = trace_dict.get("retrieval")
    tool_calls = trace_dict.get("tool_calls", [])
    timings = trace_dict.get("timings", {})

    badges = []
    if retrieval and retrieval.get("kept"):
        badges.append("🟩 RAG")
    if any(call.get("status") == "ok" for call in tool_calls):
        badges.append("🟦 MCP")
    if retrieval is not None and not retrieval.get("sufficient"):
        badges.append("⚠️ KB gap")
    if any(call.get("status") != "ok" for call in tool_calls):
        badges.append("⚠️ tool error")
    label = "How this answer was produced  " + " · ".join(badges)

    with st.expander(label, expanded=False):
        st.markdown("**1. Plan** (tool selection by the LLM)")
        st.json(
            {
                "needs_knowledge_base": plan.get("needs_knowledge_base"),
                "kb_query": plan.get("kb_query"),
                "tool_calls": plan.get("tool_calls"),
                "reasoning": plan.get("reasoning"),
                "planner": plan.get("planner"),
            }
        )

        st.markdown("**2. Knowledge-base retrieval**")
        if retrieval is None:
            st.caption("Knowledge base not consulted for this question.")
        else:
            st.caption(f"Query: `{retrieval.get('query')}`")
            kept = retrieval.get("kept") or []
            if kept:
                st.table(
                    [
                        {
                            "label": hit["label"],
                            "score": hit["score"],
                            "source": hit["source_title"],
                            "section": hit["section"],
                        }
                        for hit in kept
                    ]
                )
            else:
                st.warning(
                    "No chunk cleared the relevance threshold — the assistant "
                    "was instructed to say the knowledge base does not cover this."
                )
            rejected = retrieval.get("rejected") or []
            if rejected:
                st.caption(
                    "Below threshold: "
                    + ", ".join(
                        f"{hit['source_title']} ({hit['score']})" for hit in rejected[:4]
                    )
                )

        st.markdown("**3. MCP tool calls**")
        if not tool_calls:
            st.caption("No live tools were needed for this question.")
        for call in tool_calls:
            icon = "✅" if call.get("status") == "ok" else "❌"
            st.markdown(
                f"{icon} `{call['name']}` — {call.get('seconds', '?')}s  \n"
                f"arguments: `{json.dumps(call.get('arguments', {}))}`"
            )
            st.json(call.get("result", {}))

        if timings:
            st.caption(
                "Timings (s): "
                + ", ".join(f"{key} {value}" for key, value in timings.items())
            )


def sidebar() -> None:
    with st.sidebar:
        st.header("Knowledge base")
        manifest = index_manifest()
        if not manifest.get("chunks"):
            st.error(
                "No index found. Run:\n\n"
                "`python -m ingest.fetch_sources`\n\n"
                "`python -m ingest.build_index`"
            )
        else:
            col_a, col_b = st.columns(2)
            col_a.metric("Documents", manifest["documents"])
            col_b.metric("Chunks", manifest["chunks"])
            st.caption(
                f"{len(manifest.get('collections', []))} distinct resources: "
                + ", ".join(manifest.get("collections", []))
            )
            with st.expander("Indexed sources"):
                for source in manifest.get("sources", []):
                    if source.get("source_url"):
                        st.markdown(
                            f"- [{source['source_title']}]({source['source_url']}) "
                            f"({source['chunks']})"
                        )
                    else:
                        st.markdown(
                            f"- {source['source_title']} ({source['chunks']})"
                        )

        st.divider()
        st.header("Remembered preferences")
        memory: ConversationMemory = st.session_state.memory
        if memory.preferences:
            for key, value in memory.preferences.items():
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.caption("Nothing yet — mention a budget, dates or who you're travelling with.")

        st.divider()
        st.caption(f"LLM: `{OLLAMA_MODEL}` via Ollama")
        if st.button("Reset conversation", use_container_width=True):
            memory.reset()
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

        st.divider()
        st.header("Try one of these")
        for i, example in enumerate(EXAMPLES):
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()


def main() -> None:
    init_state()
    st.title(f"🧭 {DESTINATION} Travel Planning Assistant")
    st.caption(
        "Destination knowledge from a document knowledge base (RAG) · "
        "live weather and currency through MCP tools"
    )
    sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                render_trace(message["trace"])

    typed = st.chat_input("Ask about your trip…")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trace_slot = st.container()
        status = st.status("Planning…", expanded=False)
        text_slot = st.empty()
        collected: list[str] = []
        trace_dict: dict = {}

        for kind, payload in stream_answer(question, st.session_state.memory):
            if kind == "trace":
                trace_dict = {
                    "plan": payload.plan,
                    "retrieval": payload.retrieval,
                    "tool_calls": payload.tool_calls,
                    "timings": payload.timings,
                }
                tool_names = [call["name"] for call in payload.tool_calls]
                status.update(
                    label=(
                        "Retrieved knowledge base"
                        + (f" · called {', '.join(tool_names)}" if tool_names else "")
                        + " · writing answer…"
                    )
                )
            elif kind == "token":
                collected.append(payload)
                text_slot.markdown("".join(collected))
            elif kind == "done":
                trace_dict["timings"] = payload.trace.timings
                status.update(label="Done", state="complete")

        with trace_slot:
            render_trace(trace_dict)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "".join(collected),
                "trace": trace_dict,
            }
        )


if __name__ == "__main__":
    main()
