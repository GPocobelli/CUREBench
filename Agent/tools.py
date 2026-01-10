# tools.py
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional

import requests
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from ddgs import DDGS

# -------------------------
# Existing tools (yours)
# -------------------------

@tool("save_text_to_file", description="Saves structured research data to a text file.")
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"





@tool
def duckduckgo_search_json(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search via ddgs; returns JSON with urls."""
    with DDGS() as ddgs:
        out = []
        for r in ddgs.text(query, max_results=max_results):
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    return json.dumps(out, ensure_ascii=False)



api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


# -------------------------
# Shared HTTP helper
# -------------------------

_DEFAULT_TIMEOUT = 15

def _http_get(url: str, params=None, headers=None, max_retries: int = 6) -> requests.Response:
    h = {"User-Agent": "curebench-agent/1.0"}
    if headers:
        h.update(headers)

    base = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=h, timeout=_DEFAULT_TIMEOUT)

            # Handle rate limits / transient errors
            if resp.status_code in (429, 503, 502, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    wait = float(retry_after)
                else:
                    wait = base * (2 ** attempt) + random.uniform(0, 0.5)

                time.sleep(min(wait, 30.0))
                continue

            resp.raise_for_status()
            return resp

        except requests.RequestException:
            # network hiccup: backoff
            wait = base * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(min(wait, 30.0))

    # last try: raise
    resp.raise_for_status()
    return resp


def _truncate(text: str, max_chars: int = 2500) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


# -------------------------
# ClinicalTrials.gov API v2
# -------------------------

@tool(
    "clinicaltrials_search",
    description=(
        "Search ClinicalTrials.gov (structured trial registry). "
        "Input: free-text query. Output: list of trials (NCTId, title, status)."
    ),
)
def clinicaltrials_search(query: str, max_results: int = 3) -> str:
    """
    Search ClinicalTrials.gov v2 studies using a free-text query.
    Returns a compact list of NCTId + brief title + status.
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": query,
        "pageSize": max(1, min(int(max_results), 10)),
        "format": "json",
    }
    data = _http_get(url, params=params).json()
    studies = data.get("studies", []) or []
    out = []
    for s in studies:
        p = (s.get("protocolSection") or {})
        ident = (p.get("identificationModule") or {})
        status = (p.get("statusModule") or {})
        nct = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        overall = status.get("overallStatus", "")
        out.append({"nctId": nct, "title": title, "status": overall})
    return json.dumps(out, ensure_ascii=False)


# -------------------------
# NCBI E-utilities (PubMed)
# -------------------------

@tool("pubmed_search", description="Search PubMed via NCBI E-utilities. Input: query. Output: list of PMIDs.")
def pubmed_search(query: str, retmax: int = 5) -> str:
    """
    PubMed search -> returns PMIDs.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max(1, min(int(retmax), 20)),
    }
    data = _http_get(url, params=params).json()
    pmids = (data.get("esearchresult") or {}).get("idlist", []) or []
    return json.dumps(pmids, ensure_ascii=False)


@tool(
    "pubmed_fetch",
    description=(
        "Fetch PubMed records (abstract-level) by PMIDs. "
        "Input: comma-separated PMIDs. Output: truncated XML payload."
    ),
)
def pubmed_fetch_abstract(pmids_csv: str) -> str:
    """
    Fetch abstracts for given PMIDs (comma-separated).
    Returns a compact text payload (not full XML).
    """
    pmids = [p.strip() for p in pmids_csv.split(",") if p.strip()]
    if not pmids:
        return "[]"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids[:10]),
        "retmode": "xml",
    }
    xml_text = _http_get(url, params=params).text
    return _truncate(xml_text, 2500)


# -------------------------
# RKI RSS (Epidemiologisches Bulletin)
# -------------------------

@tool(
    "rki_epi_bulletin_latest",
    description="Get latest items from RKI Epidemiologisches Bulletin via RSS. Output: list of title/link/published.",
)
def rki_epi_bulletin_latest(max_items: int = 5) -> str:
    """
    Returns latest RKI Epidemiologisches Bulletin items via RSS.
    """
    try:
        import feedparser
    except Exception:
        return "feedparser not installed; cannot parse RSS."

    feed_url = "https://www.rki.de/SiteGlobals/Functions/RSSFeed/RSSNewsfeed/RSSNewsfeed.xml?nn=2372148"
    feed = feedparser.parse(feed_url)
    items = []
    for e in (feed.entries or [])[: max(1, min(int(max_items), 10))]:
        items.append(
            {
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
        )
    return json.dumps(items, ensure_ascii=False)


# -------------------------
# EMA RSS (e.g., EPAR / New medicines)
# -------------------------

@tool(
    "ema_rss_latest",
    description=(
        "Get latest EMA items via RSS. "
        "Input: feed_kind in {epars_human, new_medicines_human}, max_items. "
        "Output: list of title/link/published."
    ),
)
def ema_rss_latest(feed_kind: str = "epars_human", max_items: int = 5) -> str:
    """
    Fetch EMA RSS feeds. feed_kind options:
      - 'epars_human' (EPARs and summaries of opinion)
      - 'new_medicines_human'
    """
    try:
        import feedparser
    except Exception:
        return "feedparser not installed; cannot parse RSS."

    feeds = {
        "epars_human": "https://www.ema.europa.eu/en/rss?type=human_medicines_epars",
        "new_medicines_human": "https://www.ema.europa.eu/en/rss?type=human_medicines_new",
    }
    url = feeds.get(feed_kind, feeds["epars_human"])
    feed = feedparser.parse(url)

    items = []
    for e in (feed.entries or [])[: max(1, min(int(max_items), 10))]:
        items.append(
            {
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
            }
        )
    return json.dumps(items, ensure_ascii=False)


# -------------------------
# Wikidata SPARQL
# -------------------------

@tool(
    "wikidata_sparql",
    description="Query Wikidata via SPARQL. Input: SPARQL string. Output: truncated JSON results.",
)
def wikidata_sparql(query: str) -> str:
    """
    Run a SPARQL query against Wikidata Query Service and return JSON results.
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    resp = _http_get(url, params={"query": query, "format": "json"}, headers=headers).json()
    return _truncate(json.dumps(resp, ensure_ascii=False), 2500)


# -------------------------
# Open Targets GraphQL
# -------------------------

@tool(
    "opentargets_graphql",
    description="Query Open Targets GraphQL API. Input: GraphQL query + variables JSON. Output: truncated JSON.",
)
def opentargets_graphql(query: str, variables_json: str = "{}") -> str:
    """
    Query Open Targets Platform GraphQL API.
    Input: GraphQL query string and optional variables JSON.
    """
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    try:
        variables = json.loads(variables_json) if variables_json else {}
    except Exception:
        variables = {}

    payload = {"query": query, "variables": variables}
    resp = requests.post(
        url,
        json=payload,
        headers={"User-Agent": "curebench-agent/1.0"},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return _truncate(resp.text, 2500)


# -------------------------
# OPTIONAL: WHO ICD API (requires credentials)
# -------------------------

@tool(
    "icd_lookup",
    description="(Optional) ICD API lookup. Requires WHO ICD API credentials in env. Input: term. Output: status/info.",
)
def icd_lookup(term: str) -> str:
    """
    Very lightweight ICD API lookup.
    Requires WHO ICD API client id/secret and token flow; this is a stub unless configured.
    """
    client_id = os.getenv("ICD_CLIENT_ID")
    client_secret = os.getenv("ICD_CLIENT_SECRET")
    if not client_id or not client_secret:
        return "ICD API not configured (set ICD_CLIENT_ID and ICD_CLIENT_SECRET)."

    return "ICD API configured but auth flow not implemented in this minimal tool."


# -------------------------
# Export list used by the agent
# -------------------------

ALL_TOOLS = [
    duckduckgo_search_json,
    wiki_tool,
    clinicaltrials_search,      # Tool (decorated)
    pubmed_search,              # Tool (decorated)
    pubmed_fetch_abstract,       # Tool (decorated)
    rki_epi_bulletin_latest,     # Tool (decorated)
    ema_rss_latest,              # Tool (decorated)
    wikidata_sparql,             # Tool (decorated)
    opentargets_graphql,         # Tool (decorated)
    # icd_lookup,  # enable only if you really configure it
]
