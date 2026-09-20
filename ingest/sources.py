"""Declarative list of knowledge-base sources.

Three distinct public resources are used, all openly licensed and all
retrievable through stable APIs (so `fetch_sources.py` is reproducible):

  1. Wikivoyage Singapore travel guide + district articles  (CC BY-SA 4.0)
  2. Wikipedia Singapore travel-relevant articles           (CC BY-SA 4.0)
  3. Visit Singapore official pages (optional web scrape, --include-web)

Titles that no longer exist are skipped with a warning rather than failing
the whole ingest run.
"""

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# --- resource 1: Wikivoyage travel guide --------------------------------
WIKIVOYAGE_TITLES = [
    "Singapore",                        # overview, transport, itineraries, food
    "Singapore/Riverside",              # Marina Bay, museums, Merlion
    "Singapore/Orchard",                # shopping district
    "Singapore/Chinatown",              # temples, hawker centres
    "Singapore/Little India",
    "Singapore/Bugis and Kampong Glam",
    "Singapore/Sentosa",
    "Singapore/Jurong",                 # Science Centre, Gardens by the Bay West
    "Singapore/East Coast",
    "Singapore/Balestier, Newton and Toa Payoh",
    "Singapore/North and West",
    "Singapore/Changi",
]

# --- resource 2: Wikipedia practical / cultural background --------------
WIKIPEDIA_TITLES = [
    "Tourism in Singapore",
    "Mass Rapid Transit (Singapore)",
    "Public transport in Singapore",
    "Singaporean cuisine",
    "Culture of Singapore",
    "Climate of Singapore",
    "Gardens by the Bay",
    "Sentosa",
    "Singapore Botanic Gardens",
]

# --- resource 3: Visit Singapore (official tourism board) ---------------
# Scraped only when you pass --include-web. These pages are JavaScript-heavy
# and may yield little text; the ingest degrades gracefully if so.
VISIT_SINGAPORE_PAGES = [
    (
        "Visit Singapore: Essential Travel Information",
        "https://www.visitsingapore.com/travel-guide-tips/essential-travel-info/",
    ),
    (
        "Visit Singapore: Itineraries",
        "https://www.visitsingapore.com/see-do-singapore/itineraries/",
    ),
    (
        "Visit Singapore: Things to Do",
        "https://www.visitsingapore.com/see-do-singapore/",
    ),
    (
        "Visit Singapore: Getting Around",
        "https://www.visitsingapore.com/travel-guide-tips/getting-around/",
    ),
]


def wiki_url(api: str, title: str) -> str:
    """Human-readable article URL, used as the citation link."""
    host = api.replace("/w/api.php", "")
    return f"{host}/wiki/" + title.replace(" ", "_")


def all_wiki_jobs():
    """Yield (collection_name, api_endpoint, title, license) tuples."""
    for title in WIKIVOYAGE_TITLES:
        yield ("wikivoyage", WIKIVOYAGE_API, title, "CC BY-SA 4.0")
    for title in WIKIPEDIA_TITLES:
        yield ("wikipedia", WIKIPEDIA_API, title, "CC BY-SA 4.0")
