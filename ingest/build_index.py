"""Steps 2-4 of the RAG pipeline: chunk -> embed -> store in FAISS.

Chunking is two-stage:
  1. MarkdownHeaderTextSplitter keeps the document's section structure, so
     each chunk knows which heading it came from (good citations).
  2. RecursiveCharacterTextSplitter caps chunk size for the embedder.

Usage:
    python -m ingest.build_index
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_DIR,
    MANIFEST_PATH,
    RAW_DIR,
)
from rag.embeddings import get_embeddings  # noqa: E402

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
MIN_CHUNK_CHARS = 120


def parse_document(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    match = FRONT_MATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        text = text[match.end() :]
    meta.setdefault("source_title", path.stem)
    meta.setdefault("source_url", "")
    meta.setdefault("collection", "local")
    return meta, text


def chunk_document(meta: dict, text: str) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sections = header_splitter.split_text(text)
    chunks: list[Document] = []

    for section in sections:
        section_name = (
            section.metadata.get("h3")
            or section.metadata.get("h2")
            or section.metadata.get("h1")
            or "Overview"
        )
        for piece in char_splitter.split_text(section.page_content):
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Document(
                    page_content=piece.strip(),
                    metadata={
                        "source_title": meta["source_title"],
                        "source_url": meta["source_url"],
                        "collection": meta["collection"],
                        "section": section_name,
                    },
                )
            )
    return chunks


def main() -> int:
    files = sorted(RAW_DIR.glob("*.md"))
    if not files:
        print(f"No .md files in {RAW_DIR}. Run: python -m ingest.fetch_sources")
        return 1

    all_chunks: list[Document] = []
    manifest_sources = []

    for path in files:
        meta, text = parse_document(path)
        chunks = chunk_document(meta, text)
        all_chunks.extend(chunks)
        manifest_sources.append(
            {
                "file": path.name,
                "source_title": meta["source_title"],
                "source_url": meta["source_url"],
                "collection": meta["collection"],
                "chunks": len(chunks),
            }
        )
        print(f"  {path.name:<60} -> {len(chunks):>4} chunks")

    if not all_chunks:
        print("No chunks produced — check the raw documents.")
        return 1

    print(f"\nEmbedding {len(all_chunks)} chunks (first run downloads the model)…")
    embeddings = get_embeddings()

    store = FAISS.from_documents(all_chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))

    collections = sorted({s["collection"] for s in manifest_sources})
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "documents": len(files),
                "chunks": len(all_chunks),
                "collections": collections,
                "sources": manifest_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nFAISS index written to {INDEX_DIR}")
    print(f"{len(files)} documents, {len(all_chunks)} chunks, "
          f"{len(collections)} distinct resources: {', '.join(collections)}")
    print("\nNext: python scripts/test_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
