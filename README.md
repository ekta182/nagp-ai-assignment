# Singapore Travel Planning Assistant — RAG + MCP

A context-aware travel assistant that answers destination questions from a
document knowledge base (RAG) and fetches live weather and exchange rates
through MCP tools, combining both in a single weather-aware itinerary.

Runs fully locally: a local LLM via Ollama, local sentence-transformers
embeddings, a local FAISS index, and a local MCP server. No API keys.

---

## 1. Architecture

```
                         ┌─────────────────────────────┐
  user question  ───────▶│  PLANNER  (LLM, JSON mode)  │
                         │  what information is needed?│
                         └──────────┬──────────────────┘
                                    │ plan
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        ┌───────────────────────┐        ┌──────────────────────────┐
        │  RAG retrieval        │        │  MCP client (stdio)      │
        │  FAISS + BGE embed.   │        │  ├ get_current_weather   │
        │  relevance threshold  │        │  ├ get_weather_forecast  │
        │  → chunks + citations │        │  └ convert_currency      │
        └───────────┬───────────┘        └────────────┬─────────────┘
                    │  [S1..Sn] excerpts               │ JSON results
                    └───────────────┬──────────────────┘
                                    ▼
                         ┌─────────────────────────────┐
                         │  SYNTHESIS  (LLM)           │
                         │  labelling contract:        │
                         │  KB facts / live MCP / AI   │
                         └──────────┬──────────────────┘
                                    ▼
                    Streamlit chat + per-turn tool trace
```

### Why a planner instead of an autonomous agent loop

The obvious LangChain approach is `create_tool_calling_agent`, letting the
model call tools in a loop. With a 7–8B local model that is unreliable: it
picks one tool and forgets the second on combined queries, or it loops.

So tool *selection* is a single constrained-JSON LLM call (small models are
good at that), and tool *execution* is plain Python. The result is
deterministic, easy to debug, and still model-driven. A keyword heuristic in
`agent/planner.py` acts as a fallback if the JSON is unusable, plus a "safety
net" that adds a tool the keywords clearly imply but the plan omitted — the
specific failure mode of small models on combined queries.

### Components

| Path | Responsibility |
|---|---|
| `ingest/sources.py` | Declarative list of knowledge-base sources |
| `ingest/fetch_sources.py` | Download → clean → `data/raw/*.md` with metadata front matter |
| `ingest/build_index.py` | Header-aware chunking → embeddings → FAISS index + manifest |
| `rag/embeddings.py` | Local BGE embeddings, query-side instruction prefix |
| `rag/retriever.py` | Retrieval, relevance threshold, citation formatting |
| `mcp_server/travel_tools.py` | MCP server: 3 tools over 2 live capabilities |
| `agent/mcp_client.py` | MCP stdio client: tool discovery + invocation |
| `agent/planner.py` | Tool selection (LLM JSON + heuristic fallback) |
| `agent/pipeline.py` | Orchestration: plan → RAG + MCP → synthesis |
| `agent/memory.py` | Rolling transcript + durable preference store |
| `prompts/templates.py` | Both prompts, including the labelling contract |
| `app/streamlit_app.py` | Chat UI with per-turn plan/retrieval/tool trace |
| `scripts/` | Retrieval test, MCP test, CLI demo |

---

## 2. Knowledge base sources

Three distinct public resources, all fetched through stable APIs so ingestion
is reproducible:

1. **Wikivoyage Singapore travel guide** (CC BY-SA 4.0) — the main article plus
   ~11 district articles: attractions, neighbourhoods, transport, food,
   practical tips, itineraries.
2. **Wikipedia travel-relevant articles** (CC BY-SA 4.0) — MRT and public
   transport, Singaporean cuisine, culture, climate, Gardens by the Bay,
   Sentosa, Botanic Gardens, tourism overview.
3. **Visit Singapore** (official tourism board) — essential travel info,
   itineraries, things to do, getting around. Optional, enabled with
   `--include-web`; these pages are JavaScript-heavy so extraction may be thin,
   and the ingest degrades gracefully when it is.

Every document keeps `source_title`, `source_url`, `license` and
`retrieved_at` in YAML front matter. That metadata is copied onto every chunk,
so citations are real links rather than reconstructions. Content is used for
educational purposes; check each source's reuse terms before redistributing.

To add your own material, drop a `.md` file into `data/raw/` with the same
front matter and re-run `python -m ingest.build_index`.

---

## 3. RAG workflow

1. **Load** — MediaWiki API extracts and optional HTML scraping; boilerplate
   sections (references, external links, gallery) are dropped.
2. **Chunk** — two stages. `MarkdownHeaderTextSplitter` first, so each chunk
   knows its section heading; then `RecursiveCharacterTextSplitter`
   (900 chars, 150 overlap) to cap size for the embedder.
3. **Embed** — `BAAI/bge-small-en-v1.5` locally on CPU, 384 dimensions,
   L2-normalised. The BGE query-side instruction prefix is applied to queries
   only, which is how the model was trained.
4. **Store** — FAISS flat index persisted to `data/index/`, with a
   `manifest.json` that the UI reads to list indexed sources.
5. **Retrieve** — top-6 by cosine similarity. Because vectors are normalised,
   FAISS's squared-L2 distance `d` converts exactly: `cos = 1 − d/2`.
6. **Threshold** — anything below `RELEVANCE_THRESHOLD` (default 0.35) is
   discarded. If nothing survives, the synthesis prompt is told the knowledge
   base has no coverage. This is the mechanism behind honest refusals: the
   model is never handed weak chunks and asked to decline on its own.
7. **Cite** — surviving chunks are labelled `[S1]…[Sn]` with title, section and
   URL; the prompt requires inline citations and a source list.

---

## 4. MCP tools

`mcp_server/travel_tools.py` is a FastMCP server spoken to over stdio by the
official MCP Python SDK client in `agent/mcp_client.py`. Tools are discovered
at runtime via `list_tools()` and their schemas are injected into the planner
prompt — that is how they become available to the application.

| Tool | Capability | Service |
|---|---|---|
| `get_current_weather(location)` | current conditions | Open-Meteo |
| `get_weather_forecast(location, days)` | 1–16 day daily forecast | Open-Meteo |
| `convert_currency(amount, from_currency, to_currency)` | FX conversion | Frankfurter / ECB reference rates |

Neither service needs an API key.

**Failure handling.** Every tool catches its own exceptions and returns
`{"status": "error", "message": ...}` rather than raising. The client converts
timeouts and transport errors into the same shape. The synthesis prompt is
then explicitly told to report the failure and not to invent a substitute
number. `scripts/test_mcp.py` exercises both failure paths (unknown location,
unsupported currency pair).

The forecast tool also computes an `outdoor_suitability` label per day from
rain probability and precipitation, which is what drives indoor substitutions
in the combined itinerary.

**Separation of concerns.** The planner prompt forbids using MCP tools for
anything the knowledge base covers, and forbids using the knowledge base for
weather or exchange rates.

---

## 5. Prompt and context strategy

Two prompts, in `prompts/templates.py`.

**Planner prompt** — receives the live tool catalogue, the conversation
history and the stored preferences. Emits JSON only (`format="json"` on
`ChatOllama` for constrained decoding): whether the knowledge base is needed,
a self-contained `kb_query`, a list of tool calls with arguments, and any new
durable preferences. It is instructed to resolve pronouns and references
("there", "that budget") into the `kb_query`, so retrieval works on follow-up
turns where the user never repeats the destination.

**Synthesis prompt** — enforces a three-way labelling contract:

- **From the knowledge base** — destination facts only, with `[S#]` citations.
- **Live information (via MCP)** — tool name, exact figures, timestamp or rate date.
- **Suggestions** — the model's own sequencing and advice, framed as recommendations.
- **Sources** — titles and links.

Plus hard rules: say "the knowledge base does not cover this" rather than
substituting general knowledge; report tool failures instead of fabricating
numbers; never attribute a fact to a source it didn't come from; respect
stored preferences.

**Context strategy** — two kinds of memory, because they behave differently.
A rolling transcript (last 6 turns, answers truncated) lets the model resolve
references. A separate preference dict — budget, trip length, companions,
interests, dates — persists for the whole session and is injected into both
prompts, so a budget mentioned in turn 2 is still honoured in turn 9 after it
has scrolled out of the transcript window. The sidebar shows it live.

---

## 6. Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally
- ~6 GB free disk (model weights + embeddings)

### Install

```bash
git clone <your-repo-url>
cd singapore-travel-assistant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:7b-instruct    # ~4.7 GB
ollama serve                       # skip if already running as a service

cp .env.example .env               # optional, defaults are fine
```

### Build the knowledge base

```bash
python -m ingest.fetch_sources      # add --include-web to also try Visit Singapore
python -m ingest.build_index        # first run downloads the embedding model (~130 MB)
```

### Verify before running the app

```bash
python scripts/test_retrieval.py    # retrieval quality + threshold tuning
python scripts/test_mcp.py          # tool discovery, both capabilities, failure paths
```

### Run

```bash
streamlit run app/streamlit_app.py  # → http://localhost:8501
```

Or in the terminal:

```bash
python scripts/demo_cli.py             # interactive
python scripts/demo_cli.py --scripted  # full acceptance-criteria run
```

---

## 7. Configuration

Everything is in `config.py`, overridable through `.env`.

| Setting | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | best small model for JSON planning; `llama3.1:8b` also works |
| `NUM_CTX` | `8192` | **important** — Ollama defaults to 2048, which silently truncates RAG context |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | swap for `bge-base-en-v1.5` for better recall, slower |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | |
| `RETRIEVAL_K` | `6` | |
| `RELEVANCE_THRESHOLD` | `0.35` | raise if weak chunks leak in, lower if good answers get refused |
| `MAX_HISTORY_TURNS` | `6` | transcript window; preferences persist regardless |

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "No FAISS index" | Run the two ingest commands. |
| Answers ignore retrieved content | `NUM_CTX` too low, or Ollama restarted without it. |
| Everything gets refused as "not covered" | Lower `RELEVANCE_THRESHOLD`; check `scripts/test_retrieval.py` scores. |
| Off-topic chunks leak into answers | Raise `RELEVANCE_THRESHOLD`. |
| `planner: fallback` on every turn | Model isn't honouring JSON mode — pull `qwen2.5:7b-instruct`. |
| No MCP tools discovered | Run `python -m mcp_server.travel_tools` directly and read the error. |
| Tool calls return `status: error` | No internet, or Open-Meteo/Frankfurter unreachable. The app reports this rather than inventing data — which is the intended behaviour. |
| Very slow replies | CPU-only inference. Lower `ANSWER_NUM_PREDICT`, or use a smaller model. |
| Few documents fetched | Wikivoyage district titles change occasionally; the script reports which were skipped. Any three sources are enough. |

---

## 9. Acceptance criteria

| Criterion | Where |
|---|---|
| Knowledge base from ≥3 travel resources | `ingest/sources.py`; counts in the sidebar |
| Embedding-based semantic retrieval | `rag/embeddings.py`, `rag/retriever.py` |
| Grounded answers with source references | `[S#]` citations + **Sources** section |
| Weather via MCP tool | `get_current_weather`, `get_weather_forecast` |
| Currency conversion via MCP tool | `convert_currency` |
| Combined RAG + MCP response | The three-day weather-aware itinerary |
| Multi-turn conversation with retained context | `agent/memory.py`; sidebar preference panel |
| Appropriate tool selection by intent | `agent/planner.py`; visible in the turn trace |
| Missing knowledge and tool failures handled | Relevance threshold + `status: error` contract |
| Simple, usable interface | `app/streamlit_app.py` |

See `docs/sample_questions.md` for worked examples and
`docs/demo_script.md` for the recording order.
