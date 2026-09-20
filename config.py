"""Central configuration. Every module reads settings from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# --- destination ---------------------------------------------------------
DESTINATION = os.getenv("DESTINATION", "Singapore")
DESTINATION_CURRENCY = os.getenv("DESTINATION_CURRENCY", "SGD")

# --- LLM (Ollama) --------------------------------------------------------
# qwen2.5:7b-instruct is the most reliable small model for constrained JSON.
# llama3.1:8b-instruct-q4_K_M also works.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama's default context window is 2048 tokens, which silently truncates
# RAG context and makes answers look ungrounded. Always set this explicitly.
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.0"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.3"))
ANSWER_NUM_PREDICT = int(os.getenv("ANSWER_NUM_PREDICT", "512"))

# --- embeddings / retrieval ---------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE models expect this instruction on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

# Cosine similarity floor. Anything below this is treated as "the knowledge
# base does not cover this", which is what drives the honest-refusal path.
# Tune with: python scripts/test_retrieval.py
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

# --- conversation memory -------------------------------------------------
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# --- MCP server ----------------------------------------------------------
MCP_SERVER_MODULE = "mcp_server.travel_tools"
MCP_TIMEOUT_SECONDS = float(os.getenv("MCP_TIMEOUT_SECONDS", "20"))
