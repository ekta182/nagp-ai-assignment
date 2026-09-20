"""Local sentence-transformers embeddings.

BGE models are trained with an instruction prefix on the *query* side only.
Applying it to documents too (or to neither) measurably hurts retrieval, so
this thin subclass handles it in one place.

Vectors are L2-normalised at encode time, which means the FAISS squared-L2
distance d maps to cosine similarity as:  cos = 1 - d / 2
That identity is what `rag/retriever.py` uses for its relevance threshold.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


class PrefixedEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings that prepends an instruction to queries only."""

    query_prefix: str = ""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_prefix + text)


@lru_cache(maxsize=1)
def get_embeddings() -> PrefixedEmbeddings:
    prefix = BGE_QUERY_PREFIX if "bge" in EMBEDDING_MODEL.lower() else ""
    return PrefixedEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_prefix=prefix,
    )
