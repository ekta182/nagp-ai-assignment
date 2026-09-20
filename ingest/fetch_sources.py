"""Step 1 of the RAG pipeline: download source content to data/raw/*.md.

Each output file carries YAML front matter with the source title, URL and
licence. That metadata travels with every chunk so the assistant can cite
real links instead of guessing.

Usage:
    python -m ingest.fetch_sources
    python -m ingest.fetch_sources --include-web     # also try Visit Singapore
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR  # noqa: E402
from ingest.sources import (  # noqa: E402
    VISIT_SINGAPORE_PAGES,
    all_wiki_jobs,
    wiki_url,
)

USER_AGENT = (
    "TravelAssistantAssignment/1.0 (educational project; contact: student@example.com)"
)

# Wiki sections that add noise rather than travel knowledge.
DROP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "notes",
    "further reading",
    "bibliography",
    "citations",
    "gallery",
    "sources",
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:80] or "source"


def wiki_headings_to_markdown(text: str) -> str:
    """MediaWiki extracts use '== X ==' headings. Convert to markdown and
    drop the boilerplate sections."""
    out_lines: list[str] = []
    skipping = False

    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^(={2,6})\s*(.+?)\s*\1$", stripped)
        if match:
            level = len(match.group(1))          # '==' -> h2
            title = match.group(2).strip()
            skipping = title.lower() in DROP_SECTIONS
            if skipping:
                continue
            out_lines.append("")
            out_lines.append("#" * min(level, 6) + " " + title)
            out_lines.append("")
            continue

        if not skipping:
            out_lines.append(line)

    cleaned = "\n".join(out_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def fetch_wiki_article(api: str, title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": title,
    }
    resp = requests.get(
        api, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        extract = page.get("extract") or ""
        return extract if extract.strip() else None
    return None


def fetch_web_page(url: str) -> str | None:
    """Best-effort HTML extraction for the official tourism-board pages."""
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return None

    parts: list[str] = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if len(text) < 3:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            level = {"h1": "##", "h2": "##", "h3": "###", "h4": "####"}[node.name]
            parts.append(f"\n{level} {text}\n")
        elif node.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    body = "\n".join(parts).strip()
    return body if len(body) > 600 else None


def write_document(
    *, filename: str, title: str, url: str, licence: str, collection: str, body: str
) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    front_matter = (
        "---\n"
        f"source_title: {title}\n"
        f"source_url: {url}\n"
        f"license: {licence}\n"
        f"collection: {collection}\n"
        f"retrieved_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "---\n\n"
    )
    path.write_text(front_matter + f"# {title}\n\n" + body + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-web",
        action="store_true",
        help="also attempt to scrape the Visit Singapore pages",
    )
    args = parser.parse_args()

    written, skipped = 0, []

    for collection, api, title, licence in all_wiki_jobs():
        try:
            raw = fetch_wiki_article(api, title)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{title} (error: {exc})")
            continue

        if not raw:
            skipped.append(f"{title} (not found / empty)")
            continue

        body = wiki_headings_to_markdown(raw)
        if len(body) < 500:
            skipped.append(f"{title} (too short: {len(body)} chars)")
            continue

        path = write_document(
            filename=f"{collection}-{slugify(title)}.md",
            title=f"{title} ({'Wikivoyage' if collection == 'wikivoyage' else 'Wikipedia'})",
            url=wiki_url(api, title),
            licence=licence,
            collection=collection,
            body=body,
        )
        written += 1
        print(f"  saved {path.name:<60} {len(body):>7,} chars")
        time.sleep(0.4)  # be polite to the API

    if args.include_web:
        print("\nAttempting Visit Singapore pages (may fail: JS-heavy site)…")
        for title, url in VISIT_SINGAPORE_PAGES:
            try:
                body = fetch_web_page(url)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{title} (error: {exc})")
                continue
            if not body:
                skipped.append(f"{title} (no extractable text)")
                continue
            path = write_document(
                filename=f"visitsingapore-{slugify(title)}.md",
                title=title,
                url=url,
                licence="© Singapore Tourism Board — quoted for educational use",
                collection="visitsingapore",
                body=body,
            )
            written += 1
            print(f"  saved {path.name:<60} {len(body):>7,} chars")
            time.sleep(0.6)

    print(f"\n{written} documents written to {RAW_DIR}")
    if skipped:
        print(f"{len(skipped)} skipped:")
        for item in skipped:
            print(f"  - {item}")

    if written < 3:
        print(
            "\nFewer than 3 documents were fetched. Check your internet "
            "connection, then re-run. You can also drop your own .md files "
            "into data/raw/ with the same front-matter format."
        )
        return 1

    print("\nNext: python -m ingest.build_index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
