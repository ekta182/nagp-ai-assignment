# Demonstration script (~5 minutes)

Deliverable 20 asks for RAG, MCP, a combined response, and conversational
context. This order shows all four with no dead air.

**Before recording:** run `ollama serve`, `python scripts/test_mcp.py` (warms
nothing but confirms connectivity), and `streamlit run app/streamlit_app.py`.
Ask one throwaway question first so the embedding model is loaded — otherwise
your first on-camera answer includes a cold-start pause.

---

### 0. Setup shot (20s)
Show the sidebar: document count, chunk count, the list of indexed sources
with links, and the model name. States the knowledge base is real and built
from three distinct resources.

### 1. Pure RAG (45s)
> "Which neighbourhoods are best for cultural experiences?"

Expand the turn trace. Point at: `needs_knowledge_base: true`, `tool_calls: []`,
and the retrieval table with similarity scores and source sections. Then point
at the `[S#]` citations in the answer and the **Sources** section.

Says: retrieval is semantic, answers are grounded, citations are traceable.

### 2. Pure MCP — weather (30s)
> "What is the forecast for the next three days?"

Trace shows `get_weather_forecast` with its arguments and the raw JSON
result — including `rain_probability_percent` and `outdoor_suitability`. Note
the knowledge base was *not* consulted: tool selection is working in both
directions.

### 3. Pure MCP — currency (20s)
> "Convert INR 50,000 to SGD."

Trace shows `convert_currency` with the parsed amount and currency codes. The
answer quotes the ECB rate date.

### 4. Combined RAG + MCP (75s) — the centrepiece
> "Create a three-day Singapore itinerary for next week and adjust it to the
> weather forecast."

Trace shows **both** the retrieval table **and** the weather tool call in the
same turn. In the answer, point at the three labelled sections, then at the
day with the highest rain probability and its named indoor alternative.

### 5. Multi-turn context (60s)
> "I'm travelling with two kids aged 6 and 9, and my budget is INR 60,000."

Point at the sidebar preference panel updating.

> "Given that, revise the itinerary and show my budget in the local currency."

Point at: the user never repeated "Singapore", "three days", or "60,000", yet
the planner's `kb_query` is self-contained and `convert_currency` is called
with the right amount. The itinerary is now child-appropriate.

### 6. Missing knowledge (25s)
> "What are the best ski resorts in Singapore?"

Trace shows every chunk below the relevance threshold. The answer says the
knowledge base doesn't cover it and invents nothing.

### 7. Tool failure (25s)
Turn off wifi.

> "What's the weather like tomorrow?"

Trace shows `status: error`. The answer reports the tool was unavailable
rather than guessing. Turn wifi back on.

### 8. Close (20s)
One sentence on the architecture: planner → RAG + MCP → synthesis under a
labelling contract, all running locally.
