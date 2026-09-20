"""Verify retrieval quality before any LLM is involved.

Run this straight after building the index. The last query is deliberately
out of scope: its scores show you where to set RELEVANCE_THRESHOLD so that
in-scope questions pass and out-of-scope ones get an honest refusal.

    python scripts/test_retrieval.py
    python scripts/test_retrieval.py "your own query here"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RELEVANCE_THRESHOLD  # noqa: E402
from rag.retriever import get_knowledge_base, index_manifest  # noqa: E402

DEFAULT_QUERIES = [
    "must-visit attractions in Singapore",
    "neighbourhoods for cultural experiences",
    "getting around Singapore by MRT and bus",
    "activities for families with children",
    "three day sightseeing itinerary",
    "indoor attractions for a rainy day",
    "hawker centres and local food",
    "best ski resorts and snow slopes",  # out of scope on purpose
]


def main() -> int:
    manifest = index_manifest()
    print(
        f"Index: {manifest['documents']} documents, {manifest['chunks']} chunks, "
        f"resources: {', '.join(manifest.get('collections', [])) or 'n/a'}"
    )
    print(f"Relevance threshold: {RELEVANCE_THRESHOLD}\n")

    kb = get_knowledge_base()
    queries = sys.argv[1:] or DEFAULT_QUERIES

    for query in queries:
        result = kb.search(query, threshold=-1.0)  # show everything, judge manually
        print("=" * 78)
        print(f"QUERY: {query}")
        top = result.hits[0].score if result.hits else float("nan")
        verdict = "PASS" if top >= RELEVANCE_THRESHOLD else "REFUSE (no coverage)"
        print(f"top score: {top:.3f}   -> {verdict}")
        for hit in result.hits[:4]:
            preview = " ".join(hit.text.split())[:150]
            print(f"  {hit.score:.3f}  {hit.source_title} / {hit.section}")
            print(f"         {preview}…")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
