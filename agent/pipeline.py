"""The orchestrator: plan -> execute (RAG + MCP) -> synthesise.

This is where a combined response is produced: the same turn can retrieve
knowledge-base chunks and call one or more MCP tools, then hand both to a
single synthesis call under the labelling contract in prompts/templates.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

from langchain_ollama import ChatOllama

from agent.mcp_client import call_tool
from agent.memory import ConversationMemory
from agent.planner import Plan, make_plan
from config import (
    ANSWER_NUM_PREDICT,
    ANSWER_TEMPERATURE,
    DESTINATION,
    NUM_CTX,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from prompts.templates import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    NO_INFO_NOTE,
    NO_TOOLS_NOTE,
)
from rag.retriever import RetrievalResult, get_knowledge_base

SMALL_TALK_REPLY = (
    f"Hello. I can help you plan a trip to {DESTINATION} — attractions, "
    "neighbourhoods, getting around, food, and sample itineraries from my "
    "knowledge base, plus live weather forecasts and currency conversion "
    "through MCP tools.\n\nTry: *\"Create a three-day itinerary for next week "
    "and adjust it for the weather forecast.\"*"
)


@dataclass
class Trace:
    plan: dict = field(default_factory=dict)
    retrieval: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def used_rag(self) -> bool:
        return bool(self.retrieval and self.retrieval.get("kept"))

    @property
    def used_mcp(self) -> bool:
        return any(call.get("status") == "ok" for call in self.tool_calls)


@dataclass
class Answer:
    text: str
    trace: Trace
    citations: list[dict] = field(default_factory=list)
    preference_updates: dict = field(default_factory=dict)


def get_answer_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=ANSWER_TEMPERATURE,
        num_ctx=NUM_CTX,
        num_predict=ANSWER_NUM_PREDICT,
    )


def _format_tool_results(results: list[dict]) -> str:
    if not results:
        return NO_TOOLS_NOTE
    blocks = []
    for item in results:
        header = f"TOOL: {item['name']}  ARGUMENTS: {json.dumps(item['arguments'])}"
        if item.get("status") == "ok":
            blocks.append(header + "\nRESULT:\n" + json.dumps(item["result"], indent=2))
        else:
            message = (item.get("result") or {}).get("message", "unknown error")
            blocks.append(
                header
                + f"\nRESULT: FAILED — {message}\n"
                + "Report this failure to the user. Do not invent a substitute value."
            )
    return "\n\n".join(blocks)


def _retrieve(plan: Plan) -> RetrievalResult | None:
    if not plan.needs_knowledge_base:
        return None
    query = plan.kb_query or ""
    if not query.strip():
        return None
    return get_knowledge_base().search(query)


def _execute_tools(plan: Plan) -> list[dict]:
    executed = []
    for call in plan.tool_calls:
        started = time.time()
        result = call_tool(call["name"], call.get("arguments") or {})
        executed.append(
            {
                "name": call["name"],
                "arguments": call.get("arguments") or {},
                "status": result.get("status", "error"),
                "result": result,
                "seconds": round(time.time() - started, 2),
            }
        )
    return executed


def _build_messages(
    question: str,
    memory: ConversationMemory,
    retrieval: RetrievalResult | None,
    tool_results: list[dict],
) -> list[tuple[str, str]]:
    if retrieval is None:
        kb_context = "(the knowledge base was not consulted for this question)"
        kb_sources = "(none)"
    elif retrieval.is_sufficient:
        kb_context = retrieval.as_context()
        kb_sources = retrieval.as_source_list()
    else:
        kb_context = NO_INFO_NOTE
        kb_sources = "(none)"

    return [
        ("system", ANSWER_SYSTEM.format(destination=DESTINATION)),
        (
            "human",
            ANSWER_USER.format(
                question=question,
                preferences=memory.preferences_text(),
                history=memory.history_text(),
                kb_context=kb_context,
                kb_sources=kb_sources,
                tool_results=_format_tool_results(tool_results),
            ),
        ),
    ]


def _prepare(question: str, memory: ConversationMemory):
    """Shared plan/execute stage for both the blocking and streaming paths."""
    trace = Trace()

    started = time.time()
    plan = make_plan(
        question, history=memory.history_text(), preferences=memory.preferences
    )
    trace.timings["plan"] = round(time.time() - started, 2)
    trace.plan = plan.as_dict()

    preference_updates = memory.update_preferences(plan.preferences)

    if plan.source == "small_talk":
        return plan, trace, None, [], preference_updates

    started = time.time()
    retrieval = _retrieve(plan)
    trace.timings["retrieval"] = round(time.time() - started, 2)
    if retrieval is not None:
        trace.retrieval = {
            "query": retrieval.query,
            "kept": retrieval.citation_map(),
            "rejected": [
                {
                    "source_title": hit.source_title,
                    "section": hit.section,
                    "score": round(hit.score, 3),
                }
                for hit in retrieval.rejected
            ],
            "sufficient": retrieval.is_sufficient,
        }

    started = time.time()
    tool_results = _execute_tools(plan)
    trace.timings["tools"] = round(time.time() - started, 2)
    trace.tool_calls = [
        {
            "name": item["name"],
            "arguments": item["arguments"],
            "status": item["status"],
            "seconds": item["seconds"],
            "result": item["result"],
        }
        for item in tool_results
    ]

    return plan, trace, retrieval, tool_results, preference_updates


def answer_question(question: str, memory: ConversationMemory) -> Answer:
    plan, trace, retrieval, tool_results, pref_updates = _prepare(question, memory)

    if plan.source == "small_talk":
        memory.add_turn(question, SMALL_TALK_REPLY)
        return Answer(text=SMALL_TALK_REPLY, trace=trace, preference_updates=pref_updates)

    messages = _build_messages(question, memory, retrieval, tool_results)

    started = time.time()
    try:
        text = get_answer_llm().invoke(messages).content
    except Exception as exc:  # noqa: BLE001
        text = (
            "I could not reach the local language model.\n\n"
            f"Error: `{exc}`\n\n"
            "Check that Ollama is running (`ollama serve`) and that the model "
            f"`{OLLAMA_MODEL}` is pulled (`ollama pull {OLLAMA_MODEL}`)."
        )
    trace.timings["synthesis"] = round(time.time() - started, 2)

    if not isinstance(text, str):
        text = str(text)

    memory.add_turn(question, text)
    return Answer(
        text=text,
        trace=trace,
        citations=retrieval.citation_map() if retrieval and retrieval.is_sufficient else [],
        preference_updates=pref_updates,
    )


def stream_answer(
    question: str, memory: ConversationMemory
) -> Iterator[tuple[str, object]]:
    """Yields ("trace", Trace) once the plan/tool stage is done, then
    ("token", str) chunks, then ("done", Answer). Used by the Streamlit UI so
    slow local inference doesn't look frozen."""
    plan, trace, retrieval, tool_results, pref_updates = _prepare(question, memory)
    yield "trace", trace

    if plan.source == "small_talk":
        yield "token", SMALL_TALK_REPLY
        memory.add_turn(question, SMALL_TALK_REPLY)
        yield "done", Answer(
            text=SMALL_TALK_REPLY, trace=trace, preference_updates=pref_updates
        )
        return

    messages = _build_messages(question, memory, retrieval, tool_results)
    started = time.time()
    collected: list[str] = []

    try:
        for chunk in get_answer_llm().stream(messages):
            piece = chunk.content
            if not isinstance(piece, str):
                piece = str(piece)
            if piece:
                collected.append(piece)
                yield "token", piece
    except Exception as exc:  # noqa: BLE001
        message = (
            "\n\nI could not reach the local language model.\n\n"
            f"Error: `{exc}`\n\nCheck that Ollama is running and that "
            f"`{OLLAMA_MODEL}` is pulled."
        )
        collected.append(message)
        yield "token", message

    trace.timings["synthesis"] = round(time.time() - started, 2)
    text = "".join(collected)
    memory.add_turn(question, text)
    yield "done", Answer(
        text=text,
        trace=trace,
        citations=retrieval.citation_map() if retrieval and retrieval.is_sufficient else [],
        preference_updates=pref_updates,
    )
