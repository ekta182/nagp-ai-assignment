# Sample questions and expected behaviour

Run `python scripts/demo_cli.py --scripted` to produce the actual transcript,
then paste real outputs under each heading. This file documents what each
question is *for*, so a reviewer can see the coverage at a glance.

Legend: **RAG** = knowledge base only · **MCP** = live tool only ·
**RAG+MCP** = combined.

---

## 1. Destination knowledge (RAG)

### "What are the must-visit attractions in Singapore?"
- **Plan:** `needs_knowledge_base: true`, no tools.
- **Expected:** named attractions with `[S#]` citations, a **Sources** section
  with Wikivoyage/Wikipedia links, and no weather or currency content.
- **Check:** every attraction named appears in the retrieved excerpts shown in
  the turn trace.

### "Which neighbourhoods are suitable for cultural experiences?"
- **Expected:** Chinatown, Little India, Kampong Glam and similar, each cited.

### "How can a tourist travel around Singapore?"
- **Expected:** MRT, buses, taxis, EZ-Link/contactless — grounded in the
  transport documents.

### "Suggest activities for a family with children."
- **Expected:** family-appropriate attractions from the KB, with the pacing
  advice clearly under **Suggestions** rather than presented as fact.

### "What indoor attractions can I visit?"
- **Expected:** museums, malls, indoor gardens — the retrieval that later
  feeds rainy-day substitution.

---

## 2. Live information (MCP)

### "What is the weather in Singapore right now?"
- **Plan:** `get_current_weather`, no knowledge base.
- **Expected:** **Live information (via MCP)** with the tool name,
  temperature, humidity, conditions and the observation timestamp.

### "What is the forecast for the next three days?"
- **Plan:** `get_weather_forecast(location="Singapore", days=3)`.
- **Expected:** three dated rows with max/min temperature and rain
  probability, attributed to Open-Meteo.

### "Convert INR 50,000 to SGD."
- **Plan:** `convert_currency(50000, "INR", "SGD")`.
- **Expected:** converted amount, the rate, the ECB rate date, and no
  knowledge-base content.

### "How much is 200 SGD in INR?"
- **Expected:** correct direction of conversion (SGD → INR).

---

## 3. Combined RAG + MCP

### "Create a three-day Singapore itinerary for next week and adjust it to the weather forecast." (required scenario)
- **Plan:** `needs_knowledge_base: true` **and** `get_weather_forecast(days=7)`.
- **Expected:**
  - **From the knowledge base** — attractions, districts, transport, cited.
  - **Live information (via MCP)** — the daily forecast with rain probability.
  - **Day 1 / Day 2 / Day 3** — outdoor activities on the drier days, indoor
    alternatives named explicitly on the wetter ones.
  - **Suggestions** — sequencing and pacing, labelled as recommendations.
  - **Sources** — the KB links.
- **Check:** the day assigned indoor activities is actually the highest
  rain-probability day in the tool result shown in the trace.

### "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary."
- **Plan:** `convert_currency` **and** the knowledge base.
- **Expected:** budget in SGD from the tool, itinerary from the KB, and any
  per-day spending split framed as an estimate, not a cited fact.

### "Suggest outdoor attractions and replace them with indoor options if rain is expected."
- **Expected:** conditional substitution driven by the actual forecast, with
  the swap justified by the rain probability figure.

---

## 4. Multi-turn context

1. "Create a three-day Singapore itinerary."
2. "I'm travelling with two kids aged 6 and 9, and my budget is INR 60,000."
3. "Given that, revise the itinerary and show the budget in the local currency."

- **Expected at step 3:** the destination is never restated by the user, yet
  retrieval still works (the planner writes a self-contained `kb_query`); the
  itinerary becomes child-appropriate; `convert_currency` is called with
  60000 INR → SGD without the amount being repeated.
- **Check:** the sidebar preference panel shows budget, trip length and
  travelling companions.

Follow-up that tests the transcript specifically:
4. "What about the second day — is there an indoor option nearby?"
- **Expected:** resolves "the second day" against the previous answer.

---

## 5. Missing knowledge

### "What are the best ski resorts in Singapore?"
- **Expected:** a clear statement that the knowledge base does not cover this,
  with no invented facts and no fallback to general knowledge. The trace shows
  every retrieved chunk scoring below the threshold.

### "What is the visa fee for Indian passport holders?"
- **Expected:** either a cited answer if the practical-information documents
  cover it, or an honest gap. Not a guessed number.

---

## 6. Tool failure

Disconnect your network, then ask:

### "What is the forecast for the next three days?"
- **Expected:** the assistant reports that the live weather tool was
  unavailable and says what it would have used it for. No invented forecast.
- **Check:** the trace shows `status: error` with the underlying message.

### "Convert 100 XYZ to SGD."
- **Expected:** reports that the currency pair is unsupported rather than
  inventing a rate. (Works online too — `XYZ` isn't a real currency.)
