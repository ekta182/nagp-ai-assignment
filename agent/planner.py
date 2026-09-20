"""Tool selection (MCP requirements 10-12) via a constrained-JSON LLM call.

Why a planner instead of `create_tool_calling_agent`: a 7B local model is not
reliable enough to drive an autonomous agent loop — it picks one tool and
forgets the second on combined queries, or loops. Constrained JSON generation
is something small models do well, so the LLM decides *what* to call and
plain Python does the calling. Deterministic, debuggable, and it still means
tool choice is genuinely model-driven.

If JSON parsing fails, a keyword heuristic takes over so the app degrades
instead of erroring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_ollama import ChatOllama

from agent.mcp_client import tools_for_prompt
from config import (
    DESTINATION,
    DESTINATION_CURRENCY,
    NUM_CTX,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PLANNER_TEMPERATURE,
)
from prompts.templates import PLANNER_SYSTEM, PLANNER_USER

WEATHER_WORDS = re.compile(
    r"\b(weather|forecast|rain|raining|rainy|temperature|hot|humid|humidity|"
    r"sunny|storm|thunder|monsoon|climate|umbrella|outdoor|indoor)\b",
    re.I,
)
CURRENCY_WORDS = re.compile(
    r"\b(convert|conversion|exchange|rate|budget|cost|inr|sgd|usd|eur|gbp|aud|"
    r"jpy|rupee|rupees|dollar|dollars|euro|yen|pound|money|spend|₹|\$)\b",
    re.I,
)
AMOUNT_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:k\b)?", re.I)
CODE_RE = re.compile(r"\b(INR|SGD|USD|EUR|GBP|AUD|JPY|CAD|CHF|CNY|MYR|THB)\b", re.I)
DAYS_RE = re.compile(r"\b(\d{1,2})\s*[- ]?\s*day", re.I)
WORD_DAYS_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]?\s*day", re.I
)
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# A question that is *only* a currency conversion needs no destination facts.
PURE_CONVERSION_RE = re.compile(
    r"^\s*(convert|how much is|what is|whats|what's|exchange)\b(?!.*\b(itinerary|"
    r"plan|attraction|visit|do|see|activities|trip|neighbourhood|neighborhood|"
    r"food|eat|transport|stay)\b)",
    re.I,
)
DESTINATION_WORDS = re.compile(
    r"\b(itinerary|plan|planning|attraction|attractions|visit|see|do|activities|"
    r"activity|trip|neighbourhood|neighborhood|district|food|eat|hawker|museum|"
    r"transport|mrt|bus|taxi|getting around|temple|garden|zoo|family|kids|"
    r"children|culture|cultural|shopping|stay|area|indoor|outdoor|beach|tips|"
    r"etiquette|recommend|suggest)\b",
    re.I,
)
SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thankyou|ok|okay|cool|great|bye|"
    r"goodbye|good morning|good evening)\b[\s!.,]*$",
    re.I,
)


@dataclass
class Plan:
    needs_knowledge_base: bool = False
    kb_query: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    reasoning: str = ""
    source: str = "llm"  # "llm" | "fallback" | "small_talk"

    def as_dict(self) -> dict:
        return {
            "needs_knowledge_base": self.needs_knowledge_base,
            "kb_query": self.kb_query,
            "tool_calls": self.tool_calls,
            "preferences": self.preferences,
            "reasoning": self.reasoning,
            "planner": self.source,
        }


def get_planner_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=PLANNER_TEMPERATURE,
        format="json",          # constrained decoding — the key to reliability
        num_ctx=NUM_CTX,
        num_predict=400,
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _clean_preferences(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if value in (None, "", [], {}, "null", "none", "unknown"):
            continue
        out[str(key)] = value
    return out


def _valid_tool_calls(raw: Any, known: set[str]) -> list[dict]:
    calls = []
    if not isinstance(raw, list):
        return calls
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        if not name or (known and name not in known):
            continue
        args = item.get("arguments") or item.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls[:3]


def keyword_fallback(question: str) -> Plan:
    """Heuristic plan used when the model returns unusable JSON."""
    plan = Plan(source="fallback", reasoning="keyword heuristic")

    if SMALL_TALK.match(question):
        plan.source = "small_talk"
        return plan

    if WEATHER_WORDS.search(question):
        days_match = DAYS_RE.search(question)
        word_match = WORD_DAYS_RE.search(question)
        if days_match:
            days = int(days_match.group(1))
        elif word_match:
            days = WORD_NUMBERS[word_match.group(1).lower()]
        else:
            days = 3
        if re.search(r"\b(right now|currently|today'?s weather|now)\b", question, re.I):
            plan.tool_calls.append(
                {"name": "get_current_weather", "arguments": {"location": DESTINATION}}
            )
        else:
            plan.tool_calls.append(
                {
                    "name": "get_weather_forecast",
                    "arguments": {"location": DESTINATION, "days": min(days, 16)},
                }
            )

    if CURRENCY_WORDS.search(question):
        codes = [code.upper() for code in CODE_RE.findall(question)]
        amounts = [
            float(a.replace(",", ""))
            for a in AMOUNT_RE.findall(question)
            if a.strip(" ,.")
        ]
        amount = next((a for a in amounts if a >= 10), None)
        if amount and codes:
            base = codes[0]
            target = codes[1] if len(codes) > 1 else DESTINATION_CURRENCY
            if base != target:
                plan.tool_calls.append(
                    {
                        "name": "convert_currency",
                        "arguments": {
                            "amount": amount,
                            "from_currency": base,
                            "to_currency": target,
                        },
                    }
                )

    # Anything that isn't purely a live-data lookup also wants the KB.
    weather_only = bool(
        re.match(
            r"^\s*(what'?s|what is|how'?s|is|will)\b.{0,50}\b(weather|forecast|rain)\b",
            question,
            re.I,
        )
    )
    conversion_only = bool(
        PURE_CONVERSION_RE.match(question) and CURRENCY_WORDS.search(question)
    )
    has_destination_intent = bool(DESTINATION_WORDS.search(question))

    plan.needs_knowledge_base = has_destination_intent or not (
        weather_only or conversion_only
    )
    if plan.needs_knowledge_base:
        plan.kb_query = question.strip()
    return plan


def make_plan(
    question: str, history: str = "", preferences: dict | None = None
) -> Plan:
    preferences = preferences or {}

    if SMALL_TALK.match(question):
        return Plan(source="small_talk", reasoning="greeting / small talk")

    known_tools = set()
    try:
        from agent.mcp_client import list_tools

        known_tools = {tool["name"] for tool in list_tools()}
    except Exception:  # noqa: BLE001
        pass

    try:
        llm = get_planner_llm()
        messages = [
            (
                "system",
                PLANNER_SYSTEM.format(
                    destination=DESTINATION, tool_catalogue=tools_for_prompt()
                ),
            ),
            (
                "human",
                PLANNER_USER.format(
                    preferences=json.dumps(preferences) or "{}",
                    history=history or "(none)",
                    question=question,
                ),
            ),
        ]
        raw = llm.invoke(messages).content
        parsed = _extract_json(raw if isinstance(raw, str) else str(raw))
    except Exception as exc:  # noqa: BLE001
        plan = keyword_fallback(question)
        plan.reasoning = f"planner LLM unavailable ({exc}); used keyword heuristic"
        return plan

    if not parsed:
        return keyword_fallback(question)

    plan = Plan(
        needs_knowledge_base=bool(parsed.get("needs_knowledge_base")),
        kb_query=str(parsed.get("kb_query") or "").strip(),
        tool_calls=_valid_tool_calls(parsed.get("tool_calls"), known_tools),
        preferences=_clean_preferences(parsed.get("preferences")),
        reasoning=str(parsed.get("reasoning") or "").strip(),
        source="llm",
    )

    if plan.needs_knowledge_base and not plan.kb_query:
        plan.kb_query = question.strip()

    # Safety net: the model sometimes forgets the second tool on combined
    # queries. Add what the keywords clearly imply but the plan is missing.
    heuristic = keyword_fallback(question)
    planned_names = {call["name"] for call in plan.tool_calls}
    for call in heuristic.tool_calls:
        family = call["name"].startswith("get_")
        already = any(
            name.startswith("get_") == family for name in planned_names
        )
        if not already:
            plan.tool_calls.append(call)
            plan.reasoning += " (+tool added by keyword safety net)"

    if not plan.needs_knowledge_base and not plan.tool_calls:
        plan.needs_knowledge_base = heuristic.needs_knowledge_base
        plan.kb_query = plan.kb_query or heuristic.kb_query

    return plan
