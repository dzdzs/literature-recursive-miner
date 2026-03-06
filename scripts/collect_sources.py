#!/usr/bin/env python3
"""Collect paper metadata and citation edges from OpenAlex, and BibTeX from DBLP."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

OPENALEX_BASE = "https://api.openalex.org"
DBLP_SEARCH_API = "https://dblp.org/search/publ/api"
DBLP_REC_BASE = "https://dblp.org/rec"
ARXIV_QUERY_API = "https://export.arxiv.org/api/query"
OPENREVIEW_NOTES_API = "https://api.openreview.net/notes"
SERPAPI_SEARCH_API = "https://serpapi.com/search.json"
SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
USER_AGENT = "literature-recursive-miner/0.1"


@dataclass
class DblpBibTeXResult:
    bibtex: str
    status: str
    dblp_key: str


# ------------------------------
# HTTP helpers
# ------------------------------


def _http_get_text(url: str, timeout: int = 30, retries: int = 3, backoff: float = 0.8) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without error")


def _http_get_bytes(url: str, timeout: int = 30, retries: int = 3, backoff: float = 0.8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without error")


def http_get_json(url: str, timeout: int = 30, retries: int = 3, backoff: float = 0.8) -> dict[str, Any]:
    raw = _http_get_text(url=url, timeout=timeout, retries=retries, backoff=backoff)
    return json.loads(raw)


# ------------------------------
# General utilities
# ------------------------------


def normalize_title(title: str) -> str:
    lowered = title.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def inverted_index_to_text(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, token))
    if not pairs:
        return ""
    pairs.sort(key=lambda p: p[0])
    return " ".join(token for _, token in pairs)


# ------------------------------
# OpenAlex retrieval
# ------------------------------


def _openalex_url(path: str, params: dict[str, Any] | None = None) -> str:
    base = f"{OPENALEX_BASE}{path}"
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def search_openalex_with_filter(filter_expr: str, per_page: int = 5) -> list[dict[str, Any]]:
    url = _openalex_url("/works", {"filter": filter_expr, "per-page": max(1, min(per_page, 25))})
    try:
        data = http_get_json(url)
    except Exception:
        return []
    return list(data.get("results") or [])


def get_work_by_openalex_id(work_id: str) -> dict[str, Any] | None:
    canonical = work_id
    if work_id.startswith("https://openalex.org/"):
        canonical = work_id.rsplit("/", 1)[-1]
    url = _openalex_url(f"/works/{canonical}")
    try:
        return http_get_json(url)
    except Exception:
        return None


def search_openalex_by_title(title_or_query: str, per_page: int = 10) -> list[dict[str, Any]]:
    url = _openalex_url("/works", {"search": title_or_query, "per-page": max(1, min(per_page, 25))})
    try:
        data = http_get_json(url)
    except Exception:
        return []
    return list(data.get("results") or [])


def resolve_seed_to_work(seed: str) -> dict[str, Any] | None:
    seed = seed.strip()
    if not seed:
        return None

    # Direct OpenAlex ID or URL.
    if re.fullmatch(r"W\d+", seed):
        return get_work_by_openalex_id(seed)
    if "openalex.org/W" in seed:
        work_id = seed.rstrip("/").rsplit("/", 1)[-1]
        return get_work_by_openalex_id(work_id)

    doi_match = re.search(r"(10\.[0-9]{4,9}/[^\s]+)", seed, flags=re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1).rstrip(".,);]")
        doi_filter = f"doi:https://doi.org/{doi}"
        by_doi = search_openalex_with_filter(doi_filter, per_page=3)
        if by_doi:
            return by_doi[0]

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", seed, flags=re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).replace(".pdf", "")
        by_arxiv = _resolve_arxiv_id_to_openalex(arxiv_id)
        if by_arxiv:
            return by_arxiv

    # If user passed bare arXiv ID like 1706.03762 or 1706.03762v7.
    bare_arxiv = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)", seed)
    if bare_arxiv:
        by_arxiv = _resolve_arxiv_id_to_openalex(bare_arxiv.group(1))
        if by_arxiv:
            return by_arxiv

    if "openreview.net" in seed:
        forum_match = re.search(r"[?&]id=([^&#]+)", seed)
        if forum_match:
            forum_id = forum_match.group(1)
            title = _fetch_openreview_title(forum_id)
            if title:
                candidates = search_openalex_by_title(title, per_page=10)
                if candidates:
                    return max(candidates, key=lambda c: title_similarity(title, c.get("display_name") or ""))
            by_forum = search_openalex_by_title(forum_id, per_page=5)
            if by_forum:
                return by_forum[0]

    candidates = search_openalex_by_title(seed, per_page=10)
    if not candidates:
        return None

    # Prefer exact/near-exact title match, otherwise first result.
    best = max(candidates, key=lambda c: title_similarity(seed, c.get("display_name") or ""))
    return best


def resolve_seed_to_title(seed: str) -> str:
    """
    Resolve a human-readable paper title from seed without relying on OpenAlex.
    """
    seed = (seed or "").strip()
    if not seed:
        return ""

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", seed, flags=re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).replace(".pdf", "")
        title = _fetch_arxiv_title(arxiv_id)
        if title:
            return title

    bare_arxiv = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)", seed)
    if bare_arxiv:
        title = _fetch_arxiv_title(bare_arxiv.group(1))
        if title:
            return title

    if "openreview.net" in seed:
        forum_match = re.search(r"[?&]id=([^&#]+)", seed)
        if forum_match:
            title = _fetch_openreview_title(forum_match.group(1))
            if title:
                return title

    return seed


def resolve_seed_to_work_exact(seed: str, title_threshold: float = 0.95) -> dict[str, Any] | None:
    """
    Resolve a seed to OpenAlex only when confident exact match is available.
    Returns None instead of fuzzy mismatches.
    """
    seed = seed.strip()
    if not seed:
        return None

    if re.fullmatch(r"W\d+", seed):
        return get_work_by_openalex_id(seed)
    if "openalex.org/W" in seed:
        work_id = seed.rstrip("/").rsplit("/", 1)[-1]
        return get_work_by_openalex_id(work_id)

    doi_match = re.search(r"(10\.[0-9]{4,9}/[^\s]+)", seed, flags=re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1).rstrip(".,);]").lower()
        by_doi = search_openalex_with_filter(f"doi:https://doi.org/{doi}", per_page=5)
        for item in by_doi:
            item_doi = str((item.get("doi") or "")).lower()
            if item_doi.endswith(doi):
                return item

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", seed, flags=re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).replace(".pdf", "")
        return _resolve_arxiv_id_to_openalex_exact(arxiv_id=arxiv_id, title_threshold=title_threshold)

    bare_arxiv = re.fullmatch(r"(\d{4}\.\d{4,5}(?:v\d+)?)", seed)
    if bare_arxiv:
        return _resolve_arxiv_id_to_openalex_exact(arxiv_id=bare_arxiv.group(1), title_threshold=title_threshold)

    if "openreview.net" in seed:
        forum_match = re.search(r"[?&]id=([^&#]+)", seed)
        if forum_match:
            forum_id = forum_match.group(1)
            title = _fetch_openreview_title(forum_id)
            if title:
                return resolve_title_to_openalex_exact(title, title_threshold=title_threshold)

    # Seed may be a title.
    return resolve_title_to_openalex_exact(seed, title_threshold=title_threshold)


def resolve_title_to_openalex_exact(title: str, title_threshold: float = 0.95) -> dict[str, Any] | None:
    query = (title or "").strip()
    if not query:
        return None
    candidates = search_openalex_by_title(query, per_page=10)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: title_similarity(query, c.get("display_name") or ""))
    best_sim = title_similarity(query, best.get("display_name") or "")
    if best_sim < max(0.80, min(1.0, title_threshold)):
        return None
    return best


def _resolve_arxiv_id_to_openalex(arxiv_id: str) -> dict[str, Any] | None:
    title = _fetch_arxiv_title(arxiv_id)
    if title:
        candidates = search_openalex_by_title(title, per_page=10)
        if candidates:
            return max(candidates, key=lambda c: title_similarity(title, c.get("display_name") or ""))

    # Fallback search by ID text if title lookup fails.
    candidates = search_openalex_by_title(arxiv_id, per_page=10)
    if candidates:
        return candidates[0]
    return None


def _resolve_arxiv_id_to_openalex_exact(arxiv_id: str, title_threshold: float = 0.95) -> dict[str, Any] | None:
    arxiv_id = (arxiv_id or "").replace(".pdf", "").strip()
    if not arxiv_id:
        return None
    target = arxiv_id.lower()
    if target.startswith("arxiv:"):
        target = target.split(":", 1)[1]

    title = _fetch_arxiv_title(arxiv_id)
    if title:
        candidates = search_openalex_by_title(title, per_page=25)
        for candidate in candidates:
            candidate_arxiv = extract_arxiv_id(candidate).lower()
            if candidate_arxiv == target:
                return candidate
        best = max(candidates, key=lambda c: title_similarity(title, c.get("display_name") or ""), default=None)
        if best and title_similarity(title, best.get("display_name") or "") >= max(0.80, min(1.0, title_threshold)):
            return best

    # Last attempt: search by arXiv id text and require extracted arXiv id exact match.
    by_id_text = search_openalex_by_title(target, per_page=25)
    for candidate in by_id_text:
        if extract_arxiv_id(candidate).lower() == target:
            return candidate
    return None


def _fetch_arxiv_title(arxiv_id: str) -> str:
    url = f"{ARXIV_QUERY_API}?{urlencode({'id_list': arxiv_id})}"
    try:
        xml = _http_get_text(url, timeout=25, retries=2, backoff=0.6)
        root = ET.fromstring(xml)
    except Exception:
        return ""

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return ""
    title = entry.findtext("atom:title", default="", namespaces=ns)
    return re.sub(r"\s+", " ", title).strip()


def _fetch_openreview_title(note_id: str) -> str:
    url = f"{OPENREVIEW_NOTES_API}?{urlencode({'id': note_id})}"
    try:
        payload = http_get_json(url, timeout=25, retries=2, backoff=0.6)
    except Exception:
        return ""

    notes = payload.get("notes") or []
    if not notes:
        return ""
    content = notes[0].get("content") or {}
    raw_title = content.get("title")
    if isinstance(raw_title, str):
        return re.sub(r"\s+", " ", raw_title).strip()
    if isinstance(raw_title, dict):
        value = raw_title.get("value")
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
    return ""


def get_backward_reference_ids(work: dict[str, Any], max_items: int = 100) -> list[str]:
    refs = list(work.get("referenced_works") or [])
    return refs[:max_items]


def get_forward_citation_ids(work_id: str, max_items: int = 100) -> list[str]:
    params = {
        "filter": f"cites:{work_id}",
        "per-page": min(max_items, 200),
    }
    url = _openalex_url("/works", params)
    try:
        data = http_get_json(url)
    except Exception:
        return []
    results = list(data.get("results") or [])
    ids = [item.get("id") for item in results if item.get("id")]
    return ids[:max_items]


def extract_arxiv_id(work: dict[str, Any]) -> str:
    ids = work.get("ids") or {}
    arxiv = ids.get("arxiv")
    if arxiv:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", arxiv, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(".pdf", "")

    locations = work.get("locations") or []
    for loc in locations:
        for key in ("pdf_url", "landing_page_url"):
            url = loc.get(key) or ""
            match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", url, flags=re.IGNORECASE)
            if match:
                return match.group(1).replace(".pdf", "")
    return ""


def normalize_arxiv_url_to_pdf(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return candidate
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", candidate, flags=re.IGNORECASE)
    if not match:
        return candidate
    arxiv_id = match.group(1).replace(".pdf", "")
    return f"https://arxiv.org/pdf/{arxiv_id}"


def choose_preferred_paper_url(work: dict[str, Any]) -> str:
    arxiv_id = extract_arxiv_id(work)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    locations = work.get("locations") or []
    for loc in locations:
        pdf_url = loc.get("pdf_url") or ""
        landing = loc.get("landing_page_url") or ""
        source_name = ((loc.get("source") or {}).get("display_name") or "").lower()

        if "openreview" in source_name or "openreview.net" in pdf_url or "openreview.net" in landing:
            if pdf_url:
                return normalize_arxiv_url_to_pdf(pdf_url)
            if landing:
                return normalize_arxiv_url_to_pdf(landing)

    best_oa = work.get("best_oa_location") or {}
    for key in ("pdf_url", "landing_page_url"):
        url = best_oa.get(key)
        if url:
            return normalize_arxiv_url_to_pdf(url)

    primary = work.get("primary_location") or {}
    for key in ("pdf_url", "landing_page_url"):
        url = primary.get(key)
        if url:
            return normalize_arxiv_url_to_pdf(url)

    return normalize_arxiv_url_to_pdf(work.get("id") or "")


def collect_source_list(work: dict[str, Any], has_dblp: bool) -> str:
    names: set[str] = {"OpenAlex"}
    if extract_arxiv_id(work):
        names.add("arXiv")

    for loc in work.get("locations") or []:
        source_name = ((loc.get("source") or {}).get("display_name") or "").lower()
        if "openreview" in source_name:
            names.add("OpenReview")

    if has_dblp:
        names.add("DBLP")

    return "|".join(sorted(names))


# ------------------------------
# Fallback expansion sources
# ------------------------------


def _strip_html_tags(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _clean_candidate_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", (title or "")).strip()
    cleaned = re.sub(r"^\[pdf\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\|\s*semanticscholar\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\|\s*google scholar\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" -")


def _extract_reference_entries(text: str) -> list[str]:
    if not text.strip():
        return []
    match = re.search(r"(?im)^\s*(references|bibliography)\s*$", text)
    if not match:
        return []
    ref_text = text[match.end() :]
    lines = [line.strip() for line in ref_text.splitlines() if line.strip()]
    entries: list[str] = []
    current: list[str] = []
    start_pattern = re.compile(r"^(?:\[\d+\]|\d+\.\s+|•\s+|\(\d+\)\s+)")
    for line in lines:
        if start_pattern.match(line) and current:
            entries.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current))
    return entries


def _extract_title_from_reference_entry(entry: str) -> str:
    if not entry:
        return ""
    compact = re.sub(r"\s+", " ", entry).strip()
    if len(compact) > 320:
        return ""
    lowered = compact.lower()
    if lowered.startswith("table ") or lowered.startswith("figure ") or "appendix" in lowered:
        return ""

    # quoted title
    quoted = re.search(r"[\"“](.+?)[\"”]", entry)
    if quoted:
        title = re.sub(r"\s+", " ", quoted.group(1)).strip(" .")
        if len(title) >= 8:
            return title

    cleaned = re.sub(r"^(?:\[\d+\]|\d+\.\s+|\(\d+\)\s+)\s*", "", compact)
    # Common style: Authors. Title, Month Year. Venue
    year_anchor = re.search(
        r",\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*20\d{2}",
        cleaned,
        flags=re.IGNORECASE,
    )
    if year_anchor:
        prefix = cleaned[: year_anchor.start()]
        dot_idx = prefix.rfind(".")
        if dot_idx != -1:
            candidate = prefix[dot_idx + 1 :].strip(" .")
            if len(re.findall(r"[A-Za-z]{3,}", candidate)) >= 3:
                return re.sub(r"\s+", " ", candidate)

    # heuristic: choose the longest sentence-like chunk with alphabetic words.
    chunks = [c.strip(" .") for c in re.split(r"\.\s+", entry) if c.strip()]
    best = ""
    best_score = 0
    for chunk in chunks:
        score = len(re.findall(r"[A-Za-z]{3,}", chunk))
        if score > best_score:
            best_score = score
            best = chunk
    if best_score >= 4:
        return re.sub(r"\s+", " ", best)
    return ""


@lru_cache(maxsize=256)
def _pdf_url_to_text(pdf_url: str) -> str:
    url = normalize_arxiv_url_to_pdf(pdf_url)
    if not url:
        return ""
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url = f"{url}.pdf"
    try:
        raw_pdf = _http_get_bytes(url, timeout=45, retries=2, backoff=0.8)
    except Exception:
        return ""

    with tempfile.TemporaryDirectory(prefix="lrm_pdf_") as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "paper.pdf")
        txt_path = os.path.join(tmp_dir, "paper.txt")
        with open(pdf_path, "wb") as f:
            f.write(raw_pdf)
        try:
            subprocess.run(
                ["pdftotext", "-enc", "UTF-8", "-nopgbrk", pdf_path, txt_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


def _normalize_pdf_text(text: str) -> str:
    compact = text.replace("\r\n", "\n").replace("\r", "\n")
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def _extract_abstract_from_pdf_text(text: str, max_chars: int = 2200) -> str:
    if not text:
        return ""
    head = text[:20000]
    start_match = re.search(r"\babstract\b[:\s\n]*", head, flags=re.IGNORECASE)
    if not start_match:
        return ""
    tail = head[start_match.end() :]
    end_match = re.search(
        r"\n\s*(?:\d+(?:\.\d+)*)?\s*(?:introduction|keywords|1\s+introduction)\b",
        tail,
        flags=re.IGNORECASE,
    )
    abstract = tail[: end_match.start()] if end_match else tail[:max_chars]
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract[:max_chars]


def _build_pdf_text_snippet(text: str, max_chars: int = 7000) -> str:
    if not text:
        return ""
    no_refs = re.split(r"\n\s*(?:references|bibliography)\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0]
    snippet = re.sub(r"\s+", " ", no_refs).strip()
    return snippet[:max_chars]


def get_text_evidence_from_pdf(
    pdf_url: str,
    *,
    abstract_max_chars: int = 2200,
    snippet_max_chars: int = 7000,
) -> tuple[str, str]:
    raw = _pdf_url_to_text(pdf_url)
    if not raw:
        return "", ""
    normalized = _normalize_pdf_text(raw)
    abstract = _extract_abstract_from_pdf_text(normalized, max_chars=abstract_max_chars)
    snippet = _build_pdf_text_snippet(normalized, max_chars=snippet_max_chars)
    return abstract, snippet


def get_backward_reference_candidates_from_pdf(pdf_url: str, max_items: int = 100) -> list[dict[str, str]]:
    text = _pdf_url_to_text(pdf_url)
    entries = _extract_reference_entries(text)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        title = _extract_title_from_reference_entry(entry)
        if not title:
            continue
        if len(re.findall(r"[A-Za-z]{3,}", title)) < 4:
            continue
        norm = normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append({"title": _clean_candidate_title(title), "paper_url": "", "source": "PDFReferences"})
        if len(out) >= max_items:
            break
    return out


def _google_scholar_query_html(query: str, start: int = 0) -> str:
    q = quote_plus(query)
    url = f"https://scholar.google.com/scholar?hl=en&q={q}&start={start}"
    try:
        return _http_get_text(url, timeout=30, retries=1, backoff=0.3)
    except Exception:
        return ""


def _parse_scholar_results(html: str) -> list[dict[str, str]]:
    if not html:
        return []
    blocks = re.findall(r'(<div class="gs_r gs_or gs_scl"[\s\S]*?</div>\s*</div>)', html)
    if not blocks:
        blocks = re.findall(r'(<div class="gs_ri"[\s\S]*?</div>\s*</div>)', html)
    parsed: list[dict[str, str]] = []
    for block in blocks:
        title_match = re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', block, flags=re.DOTALL)
        title = _strip_html_tags(title_match.group(1)) if title_match else ""
        if title.startswith("[PDF]"):
            title = title.replace("[PDF]", "", 1).strip()
        title = _clean_candidate_title(title)
        cites_match = re.search(r'href="([^"]*scholar\?cites=[^"]+)"', block)
        cites_url = cites_match.group(1) if cites_match else ""
        if cites_url.startswith("/"):
            cites_url = f"https://scholar.google.com{cites_url}"
        if title:
            parsed.append({"title": title, "cites_url": cites_url})
    return parsed


def _google_scholar_cited_by_titles(query_title: str, max_items: int = 100) -> list[dict[str, str]]:
    search_html = _google_scholar_query_html(query_title, start=0)
    hits = _parse_scholar_results(search_html)
    if not hits:
        return []
    best = max(hits, key=lambda item: title_similarity(query_title, item.get("title") or ""))
    cites_url = best.get("cites_url") or ""
    if not cites_url:
        # Fallback: no cited-by link available, use scholar search results directly.
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in hits:
            title = _clean_candidate_title((item.get("title") or "").strip())
            norm = normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({"title": title, "paper_url": "", "source": "GoogleScholarSearch"})
            if len(out) >= max_items:
                break
        return out

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    start = 0
    while len(out) < max_items and start < 1000:
        sep = "&" if "?" in cites_url else "?"
        page_url = f"{cites_url}{sep}start={start}"
        try:
            html = _http_get_text(page_url, timeout=30, retries=1, backoff=0.3)
        except Exception:
            break
        page_hits = _parse_scholar_results(html)
        if not page_hits:
            break
        new_in_page = 0
        for item in page_hits:
            title = _clean_candidate_title((item.get("title") or "").strip())
            norm = normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({"title": title, "paper_url": "", "source": "GoogleScholar"})
            new_in_page += 1
            if len(out) >= max_items:
                break
        if new_in_page == 0:
            break
        start += 10
    return out


def _serpapi_get_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{SERPAPI_SEARCH_API}?{urlencode(params)}"
    return http_get_json(url, timeout=35, retries=2, backoff=0.5)


def _serpapi_scholar_cited_by_titles(query_title: str, max_items: int = 100) -> list[dict[str, str]]:
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        return []
    try:
        search = _serpapi_get_json(
            {
                "engine": "google_scholar",
                "q": query_title,
                "hl": "en",
                "api_key": api_key,
            }
        )
    except Exception:
        return []
    organic = list(search.get("organic_results") or [])
    if not organic:
        return []
    best = max(organic, key=lambda item: title_similarity(query_title, str(item.get("title") or "")))
    cited_by = ((best.get("inline_links") or {}).get("cited_by") or {})
    cites_id = str(cited_by.get("cites_id") or "")
    if not cites_id:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in organic:
            title = _clean_candidate_title(str(item.get("title") or "").strip())
            norm = normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({"title": title, "paper_url": "", "source": "SerpAPISearch"})
            if len(out) >= max_items:
                break
        return out

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    start = 0
    while len(out) < max_items:
        try:
            cites_payload = _serpapi_get_json(
                {
                    "engine": "google_scholar",
                    "cites": cites_id,
                    "start": str(start),
                    "hl": "en",
                    "api_key": api_key,
                }
            )
        except Exception:
            break
        results = list(cites_payload.get("organic_results") or [])
        if not results:
            break
        new_in_page = 0
        for item in results:
            title = _clean_candidate_title(str(item.get("title") or "").strip())
            norm = normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({"title": title, "paper_url": "", "source": "SerpAPI"})
            new_in_page += 1
            if len(out) >= max_items:
                break
        if new_in_page == 0:
            break
        start += 10
    return out


def _semantic_scholar_cited_by_titles(query_title: str, max_items: int = 100) -> list[dict[str, str]]:
    params = {
        "query": query_title,
        "limit": 5,
        "fields": "paperId,title,url",
    }
    search_url = f"{SEMANTIC_SCHOLAR_SEARCH_API}?{urlencode(params)}"
    try:
        search_payload = http_get_json(search_url, timeout=30, retries=2, backoff=0.5)
    except Exception:
        return []
    data = list(search_payload.get("data") or [])
    if not data:
        return []
    best = max(data, key=lambda item: title_similarity(query_title, str(item.get("title") or "")))
    paper_id = str(best.get("paperId") or "")
    if not paper_id:
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    offset = 0
    while len(out) < max_items:
        limit = min(100, max_items - len(out))
        url = (
            f"{SEMANTIC_SCHOLAR_PAPER_API}/{paper_id}/citations?"
            f"{urlencode({'offset': offset, 'limit': limit, 'fields': 'citingPaper.title,citingPaper.url'})}"
        )
        try:
            payload = http_get_json(url, timeout=30, retries=2, backoff=0.5)
        except Exception:
            break
        rows = list(payload.get("data") or [])
        if not rows:
            break
        new_in_page = 0
        for row in rows:
            citing = row.get("citingPaper") or {}
            title = _clean_candidate_title(str(citing.get("title") or "").strip())
            paper_url = str(citing.get("url") or "").strip()
            norm = normalize_title(title)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append({"title": title, "paper_url": paper_url, "source": "SemanticScholar"})
            new_in_page += 1
            if len(out) >= max_items:
                break
        if new_in_page == 0:
            break
        offset += len(rows)
    if out:
        return out

    # Fallback: if citations are unavailable, use search results directly.
    seen: set[str] = set()
    fallback: list[dict[str, str]] = []
    for item in data:
        title = _clean_candidate_title(str(item.get("title") or "").strip())
        paper_url = str(item.get("url") or "").strip()
        norm = normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        fallback.append({"title": title, "paper_url": paper_url, "source": "SemanticScholarSearch"})
        if len(fallback) >= max_items:
            break
    return fallback


def get_forward_citation_candidates(
    paper_title: str,
    provider_priority: list[str] | None = None,
    max_items: int = 100,
) -> list[dict[str, str]]:
    providers = provider_priority or ["scholar_html", "serpapi", "semantic_scholar"]
    for provider in providers:
        name = provider.strip().lower()
        if name == "scholar_html":
            rows = _google_scholar_cited_by_titles(paper_title, max_items=max_items)
        elif name == "serpapi":
            rows = _serpapi_scholar_cited_by_titles(paper_title, max_items=max_items)
        elif name in {"semantic_scholar", "s2"}:
            rows = _semantic_scholar_cited_by_titles(paper_title, max_items=max_items)
        else:
            rows = []
        if rows:
            return rows
    return []


# ------------------------------
# DBLP BibTeX retrieval (strict)
# ------------------------------


def _dblp_search_hits(title: str, max_hits: int = 10) -> list[dict[str, Any]]:
    query = quote_plus(title)
    url = f"{DBLP_SEARCH_API}?q={query}&h={max_hits}&format=json"
    try:
        data = http_get_json(url)
    except Exception:
        return []

    result = (((data.get("result") or {}).get("hits") or {}).get("hit"))
    if not result:
        return []
    if isinstance(result, dict):
        return [result]
    return [item for item in result if isinstance(item, dict)]


def _score_dblp_hit(hit: dict[str, Any], target_title: str) -> tuple[float, int, int]:
    info = hit.get("info") or {}
    candidate_title = info.get("title") or ""
    similarity = title_similarity(candidate_title, target_title)

    dblp_key = info.get("key") or ""
    venue_bonus = 0
    if "/conf/" in f"/{dblp_key}/":
        venue_bonus = 2
    elif "/journals/" in f"/{dblp_key}/":
        venue_bonus = 1

    year = int(info.get("year") or 0)
    return (similarity, venue_bonus, year)


def _pick_best_dblp_hit(title: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hits:
        return None
    ranked = sorted(hits, key=lambda h: _score_dblp_hit(h, title), reverse=True)

    # Require modest title similarity to avoid pulling unrelated BibTeX.
    top = ranked[0]
    top_sim = _score_dblp_hit(top, title)[0]
    if top_sim < 0.60:
        return None
    return top


def fetch_dblp_bibtex_by_key(dblp_key: str) -> str:
    url = f"{DBLP_REC_BASE}/{dblp_key}.bib"
    return _http_get_text(url).strip()


def resolve_bibtex_from_dblp(title: str) -> DblpBibTeXResult:
    hits = _dblp_search_hits(title, max_hits=12)
    best = _pick_best_dblp_hit(title, hits)
    if not best:
        return DblpBibTeXResult(bibtex="", status="missing_dblp", dblp_key="")

    info = best.get("info") or {}
    dblp_key = info.get("key") or ""
    if not dblp_key:
        return DblpBibTeXResult(bibtex="", status="missing_dblp", dblp_key="")

    try:
        bib = fetch_dblp_bibtex_by_key(dblp_key)
    except Exception:
        return DblpBibTeXResult(bibtex="", status="missing_dblp", dblp_key="")

    return DblpBibTeXResult(bibtex=bib, status="ok", dblp_key=dblp_key)


def get_first_affiliation_full_name(work: dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    for authorship in authorships:
        institutions = authorship.get("institutions") or []
        for institution in institutions:
            name = institution.get("display_name") or ""
            if name:
                return name
    return ""


def get_abstract_text(work: dict[str, Any]) -> str:
    return inverted_index_to_text(work.get("abstract_inverted_index"))
