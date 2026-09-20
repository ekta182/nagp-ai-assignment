"""Steps 5-7 of the RAG pipeline: retrieve, threshold, format for citation.

The relevance threshold is the mechanism behind honest refusals. If nothing
clears it, `KnowledgeBase.search()` returns an empty result and the synthesis
prompt is told the knowledge base has no coverage — rather than handing the
LLM weak chunks and hoping it declines on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_community.vectorstores import FAISS

from config import (
    INDEX_DIR,
    MANIFEST_PATH,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_K,
)
from rag.embeddings import get_embeddings


@dataclass
class Hit:
    text: str
    source_title: str
    source_url: str
    section: str
    score: float


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    rejected: list[Hit] = field(default_factory=list)

    @property
    def is_sufficient(self) -> bool:
        return bool(self.hits)

    def as_context(self) -> str:
        """Numbered context block. The [S1] labels let the LLM cite inline."""
        if not self.hits:
            return "(no relevant knowledge-base content found)"
        blocks = []
        for i, hit in enumerate(self.hits, start=1):
            blocks.append(
                f"[S{i}] {hit.source_title} — section: {hit.section}\n{hit.text}"
            )
        return "\n\n".join(blocks)

    def as_source_list(self) -> str:
        seen: dict[str, str] = {}
        for hit in self.hits:
            seen.setdefault(hit.source_title, hit.source_url)
        return "\n".join(
            f"- {title}" + (f" — {url}" if url else "") for title, url in seen.items()
        )

    def citation_map(self) -> list[dict]:
        return [
            {
                "label": f"S{i}",
                "source_title": hit.source_title,
                "source_url": hit.source_url,
                "section": hit.section,
                "score": round(hit.score, 3),
            }
            for i, hit in enumerate(self.hits, start=1)
        ]


class KnowledgeBase:
    def __init__(self, store: FAISS):
        self.store = store

    @classmethod
    def load(cls) -> "KnowledgeBase":
        if not (INDEX_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index at {INDEX_DIR}. Run:\n"
                "  python -m ingest.fetch_sources\n"
                "  python -m ingest.build_index"
            )
        store = FAISS.load_local(
            str(INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # our own locally built index
        )
        return cls(store)

    def search(
        self, query: str, k: int | None = None, threshold: float | None = None
    ) -> RetrievalResult:
        k = k or RETRIEVAL_K
        threshold = RELEVANCE_THRESHOLD if threshold is None else threshold

        raw = self.store.similarity_search_with_score(query, k=k)
        result = RetrievalResult(query=query)

        for doc, distance in raw:
            # normalised vectors + squared L2  ->  cosine similarity
            cosine = 1.0 - (float(distance) / 2.0)
            hit = Hit(
                text=doc.page_content,
                source_title=doc.metadata.get("source_title", "Unknown source"),
                source_url=doc.metadata.get("source_url", ""),
                section=doc.metadata.get("section", ""),
                score=cosine,
            )
            (result.hits if cosine >= threshold else result.rejected).append(hit)

        return result


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load()


def index_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"documents": 0, "chunks": 0, "collections": [], "sources": []}
