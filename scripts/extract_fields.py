#!/usr/bin/env python3
"""Field extraction and normalization for literature-recursive-miner."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from collect_sources import extract_arxiv_id, get_first_affiliation_full_name


def _safe_int(text: str, default: int) -> int:
    try:
        return int(text)
    except Exception:
        return default


def _format_year_month(year: int, month: int) -> str:
    year = max(1900, min(2100, year))
    month = max(1, min(12, month))
    return f"{year:04d}/{month:02d}"


def infer_publish_time(work: dict[str, Any]) -> str:
    """Prefer arXiv ID month/year. Fall back to OpenAlex publication_date."""
    arxiv_id = extract_arxiv_id(work)
    if arxiv_id:
        # New-style IDs usually begin with YYMM.
        match = re.match(r"(?P<yy>\d{2})(?P<mm>\d{2})\.", arxiv_id)
        if match:
            yy = _safe_int(match.group("yy"), 0)
            mm = _safe_int(match.group("mm"), 1)
            year = 2000 + yy if yy <= 30 else 1900 + yy
            return _format_year_month(year, mm)

    publication_date = work.get("publication_date") or ""
    if publication_date:
        try:
            dt = datetime.strptime(publication_date, "%Y-%m-%d")
            return _format_year_month(dt.year, dt.month)
        except ValueError:
            pass

    year = _safe_int(str(work.get("publication_year") or 0), 0)
    if year > 0:
        return _format_year_month(year, 1)

    return ""


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_method_or_bench_name(title: str, abstract: str) -> str:
    """
    Prefer explicit method/benchmark names in abstract, otherwise title keywords.
    Preserve original casing from extracted phrase.
    """
    abstract = abstract or ""
    title = title or ""

    patterns = [
        r"(?:we|this paper)\s+(?:introduce|propose|present)\s+(?:a|an|the)?\s*([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,5})",
        r"(?:method|framework|benchmark|suite|environment)\s+(?:called|named)\s+([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,5})",
        r"\b([A-Z]{2,}[A-Za-z0-9\-]*)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, abstract)
        if match:
            candidate = _clean_spaces(match.group(1))
            if len(candidate) >= 2:
                return candidate

    # Fallback: keep title phrase with likely method markers.
    title_markers = [
        r"([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,4}\s+(?:Benchmark|Suite|Environment|Agent))",
        r"((?:[A-Z][a-z0-9]+\s+){1,5}(?:for|with)\s+(?:[A-Z][a-z0-9]+\s*){1,3})",
    ]
    for pattern in title_markers:
        match = re.search(pattern, title)
        if match:
            return _clean_spaces(match.group(1))

    # Final fallback: title itself trimmed to a concise phrase.
    return _clean_spaces(title[:120])


def build_alias(title: str, method_or_bench_name: str, index: int) -> str:
    """
    Prefer explicit method/benchmark-style alias (e.g., MAKG from "MAKG: ...").
    Fall back to acronymized method name, then title initials with index.
    """
    title = (title or "").strip()
    method_or_bench_name = (method_or_bench_name or "").strip()

    # 1) Strongest signal: leading token before ":" in title (e.g., "MAKG: ...")
    lead = re.match(r"^\s*([A-Za-z][A-Za-z0-9\-]{1,32})\s*[:：]\s*", title)
    if lead:
        return lead.group(1)

    # 2) Uppercase acronym token in extracted method/bench name.
    acronym_token = re.search(r"\b([A-Z][A-Z0-9\-]{1,24})\b", method_or_bench_name)
    if acronym_token:
        return acronym_token.group(1)

    # 3) Concise extracted method/bench phrase.
    words = re.findall(r"[A-Za-z0-9]+", method_or_bench_name)
    if 1 <= len(words) <= 4 and len(method_or_bench_name) <= 40:
        return method_or_bench_name

    # 4) Acronymize extracted method/bench phrase.
    if 2 <= len(words) <= 12:
        initials = "".join(word[0].upper() for word in words)
        if len(initials) >= 2:
            return initials

    # 5) Final fallback: title initials + stable index.
    title_words = re.findall(r"[A-Za-z0-9]+", title)
    fallback = "".join(word[0].upper() for word in title_words[:6])
    if not fallback:
        fallback = "PAPER"
    return f"{fallback}-{index:04d}"


def build_record(
    *,
    main_task: str,
    work: dict[str, Any],
    paper_alias: str,
    paper_url: str,
    work_type_labels: list[str],
    bibtex: str,
    bibtex_status: str,
    method_or_bench_name: str,
    source_used: str,
    parent_seed: str,
) -> dict[str, Any]:
    title = (work.get("display_name") or "").strip()
    return {
        "main_task": main_task,
        "paper_alias": paper_alias,
        "paper_title": title,
        "paper_url": paper_url,
        "work_type": "|".join(work_type_labels),
        "first_author_affiliation": get_first_affiliation_full_name(work),
        "publish_time": infer_publish_time(work),
        "bibtex": bibtex,
        "bibtex_status": bibtex_status,
        "method_or_bench_name": method_or_bench_name,
        "source_used": source_used,
        "parent_seed": parent_seed,
    }
