#!/usr/bin/env python3
"""Run recursive literature mining pipeline."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classify_paper import classify_relevance_with_llm, classify_with_llm, has_openai_api_key
from collect_sources import (
    collect_source_list,
    choose_preferred_paper_url,
    get_abstract_text,
    get_backward_reference_candidates_from_pdf,
    get_backward_reference_ids,
    get_forward_citation_candidates,
    get_forward_citation_ids,
    get_work_by_openalex_id,
    normalize_title,
    resolve_seed_to_title,
    resolve_seed_to_work_exact,
    resolve_bibtex_from_dblp,
    resolve_seed_to_work,
    resolve_title_to_openalex_exact,
    normalize_arxiv_url_to_pdf,
    get_text_evidence_from_pdf,
)
from export_csv import CSV_COLUMNS, DROPPED_COLUMNS, write_work_type_stats_csv
from extract_fields import build_alias, build_record, extract_method_or_bench_name


@dataclass
class QueueNode:
    key: str
    openalex_id: str
    title: str
    paper_url: str
    edge_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive literature miner")
    parser.add_argument(
        "--mode",
        choices=["assistant-run", "local-llm", "collect-only"],
        default="assistant-run",
        help=(
            "assistant-run/local-llm: real-time LLM filtering+classification (default); "
            "collect-only: no LLM calls, export candidates only"
        ),
    )
    parser.add_argument("--topic", required=True, help="Main task/topic for output metadata")
    parser.add_argument("--theme", required=True, help="Runtime relevance theme (required every run)")
    parser.add_argument("--keyword", action="append", default=[], help="Keyword (repeatable)")
    parser.add_argument(
        "--filter-condition",
        action="append",
        default=[],
        help="Additional runtime filter condition (repeatable), e.g. multi-turn environment required",
    )
    parser.add_argument(
        "--relevance-instruction",
        default="",
        help="Runtime natural-language relevance rule. If omitted, auto-built from theme/keyword/filter-condition.",
    )
    parser.add_argument("--seed", action="append", required=True, help="Seed paper title/identifier")
    parser.add_argument(
        "--work-type-instruction",
        default="",
        help="Natural-language instruction for runtime work_type classification (required for real-time LLM modes)",
    )
    parser.add_argument("--max-papers", type=int, default=300)
    parser.add_argument(
        "--max-processed",
        type=int,
        default=0,
        help="Optional cap on processed papers (kept+dropped). 0 means no cap.",
    )
    parser.add_argument("--max-related-per-paper", type=int, default=100)
    parser.add_argument(
        "--citation-provider-priority",
        default="scholar_html,serpapi,semantic_scholar",
        help="Provider priority for forward citations in fallback mode.",
    )
    parser.add_argument(
        "--prefer-openalex-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If exact OpenAlex match exists, prefer OpenAlex graph expansion.",
    )
    parser.add_argument(
        "--disable-openalex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable all OpenAlex resolution and graph expansion; use PDF/Scholar expansion only.",
    )
    parser.add_argument(
        "--openalex-title-exact-threshold",
        type=float,
        default=0.95,
        help="Exact-match threshold for title-based OpenAlex resolution.",
    )
    parser.add_argument("--topic-keyword", action="append", default=[], help="Coarse enqueue topic keyword")
    parser.add_argument("--agent-keyword", action="append", default=[], help="Coarse enqueue agent keyword")
    parser.add_argument(
        "--agentic-rag-keyword",
        action="append",
        default=[],
        help="Coarse enqueue keyword for agentic RAG signals",
    )
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-key", default="", help="Optional runtime API key override")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", ""))
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--output", default="papers.csv")
    parser.add_argument("--dropped-output", default="dropped.csv")
    parser.add_argument("--stats-output", default="work_type_stats.csv")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _as_openalex_id(work: dict) -> str:
    work_id = str(work.get("id") or "")
    if work_id.startswith("https://openalex.org/"):
        return work_id.rsplit("/", 1)[-1]
    return work_id


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _build_relevance_instruction(args: argparse.Namespace) -> str:
    if args.relevance_instruction.strip():
        return args.relevance_instruction.strip()
    conditions = "; ".join(args.filter_condition) if args.filter_condition else "none"
    keywords = ", ".join(args.keyword) if args.keyword else "none"
    return (
        f"Keep papers relevant to theme '{args.theme}'. "
        f"Prioritize keyword alignment with: {keywords}. "
        f"Additional constraints: {conditions}. "
        "Treat multi-turn as an annotation signal, not a hard exclusion rule. "
        "If a paper is relevant to the theme/keywords but not clearly multi-turn, keep it and mark the signal accordingly. "
        "If evidence is insufficient but potentially relevant, keep it and mark rationale with uncertain_keep. "
        "Return clearly irrelevant papers as not relevant."
    )


def _annotate_multi_turn_signal(signal: str, *, kept: bool) -> str:
    normalized = (signal or "").strip().lower()
    if normalized not in {"yes", "no", "unclear"}:
        normalized = "unclear"
    if not kept:
        return normalized
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no (included)"
    return "unclear (included)"


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.mode in {"assistant-run", "local-llm"}:
        if not args.work_type_instruction.strip():
            raise RuntimeError("--work-type-instruction is required in real-time LLM mode")
        if not has_openai_api_key():
            raise RuntimeError("OPENAI_API_KEY (or API_KEY / --api-key) is required in real-time LLM mode")


def _apply_runtime_api_config(args: argparse.Namespace) -> None:
    if args.api_key.strip():
        os.environ["OPENAI_API_KEY"] = args.api_key.strip()
    if args.api_base_url.strip():
        os.environ["API_BASE_URL"] = args.api_base_url.strip()
    if args.llm_model.strip():
        os.environ["OPENAI_MODEL"] = args.llm_model.strip()


def _open_live_writer(path: str, columns: list[str]) -> tuple[Path, Any, csv.DictWriter]:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    fh.flush()
    return out_path, fh, writer


def _write_live_row(fh: Any, writer: csv.DictWriter, row: dict, columns: list[str]) -> None:
    writer.writerow({col: row.get(col, "") for col in columns})
    fh.flush()


def _make_unique_alias(alias: str, used_aliases: set[str], index: int) -> str:
    candidate = (alias or "").strip()
    if not candidate:
        candidate = f"PAPER-{index:04d}"
    if candidate not in used_aliases:
        used_aliases.add(candidate)
        return candidate

    suffix = 2
    while True:
        with_suffix = f"{candidate}-{suffix:02d}"
        if with_suffix not in used_aliases:
            used_aliases.add(with_suffix)
            return with_suffix
        suffix += 1


def _normalize_seed_url(seed: str) -> str:
    return normalize_arxiv_url_to_pdf((seed or "").strip())


def _normalize_priority(priority_csv: str) -> list[str]:
    items = [item.strip().lower() for item in (priority_csv or "").split(",")]
    return [item for item in items if item]


def _normalize_keywords(raw: list[str], defaults: list[str]) -> list[str]:
    merged = [item.strip().lower() for item in raw if item.strip()]
    if merged:
        return merged
    return defaults


def _has_any_term(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _passes_enqueue_coarse_filter(
    *,
    title: str,
    topic_terms: list[str],
    agent_terms: list[str],
    agentic_rag_terms: list[str],
) -> bool:
    lowered = (title or "").strip().lower()
    if not lowered:
        return False
    has_topic = _has_any_term(lowered, topic_terms)
    has_agent = _has_any_term(lowered, agent_terms)
    mentions_rag = ("rag" in lowered) or ("retrieval augmented generation" in lowered)
    if mentions_rag:
        has_agentic_rag = _has_any_term(lowered, agentic_rag_terms) or has_agent
    else:
        has_agentic_rag = True
    # Coarse filter should be permissive to avoid missing candidates.
    # Keep if topic-related OR agent-related; for RAG still require agentic signal.
    return (has_topic or has_agent) and has_agentic_rag


def _node_key_from_openalex_id(openalex_id: str) -> str:
    return f"OA:{openalex_id}"


def _node_key_from_title(title: str) -> str:
    return f"T:{normalize_title(title)}"


def _is_news_report_title(title: str) -> bool:
    lowered = (title or "").strip().lower()
    if not lowered:
        return False
    patterns = [
        "introducing ",
        " now available",
        "available on ",
        " announcement",
        "press release",
        " blog",
        " | ",
        "openai",
        "gemini",
        "perplexity",
    ]
    if any(pat in lowered for pat in patterns):
        return True
    # Examples: "... , April 2025"
    if any(month in lowered for month in ["january", "february", "march", "april", "may", "june", "july"]):
        if re.search(r"\b20\d{2}\b", lowered):
            return True
    return False


def _is_uncertain_reason_text(reason: str) -> bool:
    lowered = (reason or "").strip().lower()
    if not lowered:
        return False
    patterns = [
        "unclear",
        "uncertain",
        "insufficient",
        "unable to confirm",
        "no abstract",
        "lack evidence",
        "缺少",
        "无法确认",
        "不确定",
        "证据不足",
        "未提供摘要",
    ]
    return any(pat in lowered for pat in patterns)


def _has_search_signal(title: str, topic_terms: list[str]) -> bool:
    lowered = (title or "").strip().lower()
    search_terms = [
        "deep research",
        "search",
        "information seeking",
        "retrieval",
        "question answering",
        "web",
        "research",
    ]
    merged = list(dict.fromkeys([*topic_terms, *search_terms]))
    return any(term in lowered for term in merged)


def _has_agent_signal(title: str, agent_terms: list[str]) -> bool:
    lowered = (title or "").strip().lower()
    merged = list(dict.fromkeys([*agent_terms, "agent", "assistant", "multi-agent", "workflow", "benchmark", "gaia"]))
    return any(term in lowered for term in merged)


def _should_keep_as_uncertain(
    *,
    title: str,
    rationale: str,
    topic_terms: list[str],
    agent_terms: list[str],
) -> bool:
    lowered = (title or "").strip().lower()
    has_search = _has_search_signal(lowered, topic_terms)
    has_agent = _has_agent_signal(lowered, agent_terms)
    news_like = _is_news_report_title(lowered)
    uncertain_reason = _is_uncertain_reason_text(rationale)

    # Keep deep-research/search announcements and reports.
    if news_like and has_search:
        return True
    # Keep agent benchmark style items (e.g., GAIA) when search relevance is uncertain.
    if has_agent and ("benchmark" in lowered or "gaia" in lowered):
        return True
    # General conservative fallback: uncertain evidence but has thematic signal.
    if uncertain_reason and (has_search or has_agent):
        return True
    return False


def _normalize_work_type_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean = str(label or "").strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _resolve_node_to_work(node: QueueNode, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if args.disable_openalex:
        return None, ""
    if node.openalex_id:
        work = get_work_by_openalex_id(node.openalex_id)
        if work:
            return work, node.openalex_id
    if not node.title:
        return None, ""
    if not args.prefer_openalex_exact:
        return None, ""
    exact = resolve_title_to_openalex_exact(node.title, title_threshold=args.openalex_title_exact_threshold)
    if not exact:
        return None, ""
    return exact, _as_openalex_id(exact)


def run() -> int:
    args = parse_args()
    _apply_runtime_api_config(args)
    _validate_mode_args(args)

    main_task = args.topic.strip()
    if args.keyword:
        main_task = f"{main_task} | keywords: {', '.join(args.keyword)}"
    relevance_instruction = _build_relevance_instruction(args)

    provider_priority = _normalize_priority(args.citation_provider_priority)
    topic_defaults = [*args.keyword, "deep research", "research", "search", "information seeking"]
    topic_terms = _normalize_keywords(args.topic_keyword, topic_defaults)
    agent_terms = _normalize_keywords(
        args.agent_keyword,
        ["agent", "agents", "agentic", "assistant", "autonomous", "workflow"],
    )
    agentic_rag_terms = _normalize_keywords(
        args.agentic_rag_keyword,
        [
            "agentic rag",
            "tool use",
            "tool-use",
            "planning",
            "workflow",
            "orchestration",
            "action loop",
            "multi-step",
            "react",
        ],
    )

    queue: deque[str] = deque()
    node_by_key: dict[str, QueueNode] = {}
    enqueued: set[str] = set()
    visited: set[str] = set()
    seen_titles: set[str] = set()
    used_aliases: set[str] = set()
    root_seed_by_node: dict[str, str] = {}

    records: list[dict] = []
    dropped_records: list[dict] = []
    output_path, records_fh, records_writer = _open_live_writer(args.output, CSV_COLUMNS)
    dropped_path, dropped_fh, dropped_writer = _open_live_writer(args.dropped_output, DROPPED_COLUMNS)

    try:
        def enqueue_node(
            *,
            title: str,
            openalex_id: str,
            paper_url: str,
            edge_source: str,
            parent_seed: str,
            apply_coarse_filter: bool,
        ) -> bool:
            clean_title = (title or "").strip()
            clean_openalex = (openalex_id or "").strip()
            clean_url = _normalize_seed_url(paper_url or "")

            if clean_openalex and not clean_title:
                fetched = get_work_by_openalex_id(clean_openalex)
                clean_title = str((fetched or {}).get("display_name") or "").strip()
                if fetched and not clean_url:
                    clean_url = choose_preferred_paper_url(fetched)

            if not clean_title and not clean_openalex:
                return False

            if apply_coarse_filter:
                probe = clean_title or clean_openalex
                if not _passes_enqueue_coarse_filter(
                    title=probe,
                    topic_terms=topic_terms,
                    agent_terms=agent_terms,
                    agentic_rag_terms=agentic_rag_terms,
                ):
                    return False

            key = _node_key_from_openalex_id(clean_openalex) if clean_openalex else _node_key_from_title(clean_title)
            if key in enqueued or key in visited:
                return False
            node_by_key[key] = QueueNode(
                key=key,
                openalex_id=clean_openalex,
                title=clean_title,
                paper_url=clean_url,
                edge_source=edge_source,
            )
            queue.append(key)
            enqueued.add(key)
            if parent_seed and key not in root_seed_by_node:
                root_seed_by_node[key] = parent_seed
            return True

        for seed in args.seed:
            seed_parent = _normalize_seed_url(seed)
            if args.disable_openalex:
                seed_title = resolve_seed_to_title(seed) or seed
                enqueue_node(
                    title=seed_title,
                    openalex_id="",
                    paper_url=seed_parent,
                    edge_source="seed_fallback",
                    parent_seed=seed_parent,
                    apply_coarse_filter=False,
                )
                continue

            work = (
                resolve_seed_to_work_exact(seed, title_threshold=args.openalex_title_exact_threshold)
                if args.prefer_openalex_exact
                else resolve_seed_to_work(seed)
            )
            if work:
                work_id = _as_openalex_id(work)
                title = str(work.get("display_name") or "").strip()
                paper_url = choose_preferred_paper_url(work)
                if not enqueue_node(
                    title=title,
                    openalex_id=work_id,
                    paper_url=paper_url,
                    edge_source="openalex_exact" if args.prefer_openalex_exact else "openalex_resolved",
                    parent_seed=seed_parent,
                    apply_coarse_filter=False,
                ):
                    _log(args.verbose, f"[WARN] Seed not enqueued (duplicate/invalid): {seed}")
                continue

            # Exact OpenAlex not found: keep as title/url seed node for fallback expansion.
            enqueue_node(
                title=seed,
                openalex_id="",
                paper_url=seed_parent,
                edge_source="seed_fallback",
                parent_seed=seed_parent,
                apply_coarse_filter=False,
            )
            _log(args.verbose, f"[WARN] Seed not exactly resolved via OpenAlex, using fallback node: {seed}")

        if not queue:
            raise RuntimeError("No seed papers could be resolved.")

        index = 1
        while queue and len(records) < args.max_papers:
            if args.max_processed > 0 and (len(records) + len(dropped_records)) >= args.max_processed:
                _log(
                    args.verbose,
                    f"[INFO] Reached max_processed={args.max_processed}; stopping traversal early.",
                )
                break
            node_key = queue.popleft()
            if node_key in visited:
                continue
            visited.add(node_key)
            node = node_by_key.get(node_key)
            if not node:
                continue

            work, resolved_openalex_id = _resolve_node_to_work(node, args)
            if work and resolved_openalex_id and not node.openalex_id:
                node.openalex_id = resolved_openalex_id
                node.edge_source = "openalex_exact"
                if not node.title:
                    node.title = str(work.get("display_name") or "")
                if not node.paper_url:
                    node.paper_url = choose_preferred_paper_url(work)
                node_by_key[node_key] = node

            title = str((work or {}).get("display_name") or node.title or "").strip()
            if not title:
                continue
            current_paper_url = _normalize_seed_url(node.paper_url)

            normalized = normalize_title(title)
            if normalized in seen_titles:
                _log(args.verbose, f"[INFO] Skip duplicate title: {title}")
            else:
                seen_titles.add(normalized)
                abstract = get_abstract_text(work) if work else ""

                if work:
                    paper_url = choose_preferred_paper_url(work)
                    source_used = collect_source_list(work, has_dblp=False)
                else:
                    paper_url = _normalize_seed_url(node.paper_url)
                    source_used = node.edge_source or "PDFReferences|Scholar"

                if not paper_url and node.paper_url:
                    paper_url = _normalize_seed_url(node.paper_url)
                current_paper_url = paper_url

                full_text_snippet = ""
                if not abstract and current_paper_url:
                    pdf_abstract, pdf_snippet = get_text_evidence_from_pdf(current_paper_url)
                    if pdf_abstract:
                        abstract = pdf_abstract
                    if pdf_snippet:
                        full_text_snippet = pdf_snippet

                method_text = abstract or full_text_snippet
                method_name = extract_method_or_bench_name(title=title, abstract=method_text)
                alias_base = build_alias(title=title, method_or_bench_name=method_name, index=index)
                alias = _make_unique_alias(alias_base, used_aliases, index=index)
                index += 1

                if not work:
                    work = {
                        "display_name": title,
                        "id": paper_url or node_key,
                        "authorships": [],
                        "locations": [],
                        "publication_date": "",
                        "publication_year": "",
                    }

                bibtex_result = resolve_bibtex_from_dblp(title)
                record_sources = [item for item in source_used.split("|") if item]
                if bibtex_result.status == "ok" and "DBLP" not in record_sources:
                    record_sources.append("DBLP")
                if full_text_snippet and "PDFText" not in record_sources:
                    record_sources.append("PDFText")
                work_type_labels: list[str] = []
                multi_turn_signal = "unreviewed"
                relevance_rationale = "collect_only_pending"

                if args.mode in {"assistant-run", "local-llm"}:
                    relevance_result = classify_relevance_with_llm(
                        title=title,
                        abstract=abstract,
                        theme=args.theme,
                        keywords=args.keyword,
                        filter_conditions=args.filter_condition,
                        relevance_instruction=relevance_instruction,
                        model=args.llm_model,
                        full_text_snippet=full_text_snippet,
                    )
                    if relevance_result.get("status") != "ok":
                        raise RuntimeError(
                            f"Relevance classification failed for '{title}': {relevance_result.get('rationale')}"
                        )

                    is_relevant = bool(relevance_result.get("is_relevant"))
                    multi_turn_signal = str(relevance_result.get("multi_turn_signal", "unclear"))
                    relevance_rationale = str(relevance_result.get("rationale", ""))

                    if not is_relevant and _should_keep_as_uncertain(
                        title=title,
                        rationale=relevance_rationale,
                        topic_terms=topic_terms,
                        agent_terms=agent_terms,
                    ):
                        is_relevant = True
                        relevance_rationale = (
                            "uncertain_keep: retained by conservative policy because the item is "
                            "potentially relevant to agent/search capabilities despite incomplete evidence. "
                            f"original_judgement={relevance_rationale}"
                        ).strip()

                    if not is_relevant:
                        dropped_row = {
                            "main_task": main_task,
                            "theme": args.theme,
                            "paper_alias": alias,
                            "paper_title": title,
                            "paper_url": paper_url,
                            "parent_seed": root_seed_by_node.get(node_key, ""),
                            "drop_reason": "not_relevant",
                            "multi_turn_signal": _annotate_multi_turn_signal(multi_turn_signal, kept=False),
                            "relevance_rationale": relevance_rationale,
                        }
                        dropped_records.append(dropped_row)
                        _write_live_row(dropped_fh, dropped_writer, dropped_row, DROPPED_COLUMNS)
                        _log(args.verbose, f"[DROP] {title}")
                        print(
                            f"[PROGRESS] processed={len(records) + len(dropped_records)} "
                            f"kept={len(records)} dropped={len(dropped_records)} queue={len(queue)}"
                        )
                        if args.sleep_seconds > 0:
                            time.sleep(args.sleep_seconds)
                        continue

                    labels_result = classify_with_llm(
                        title=title,
                        abstract=abstract,
                        work_type_instruction=args.work_type_instruction,
                        model=args.llm_model,
                        full_text_snippet=full_text_snippet,
                    )
                    if labels_result.get("status") != "ok":
                        raise RuntimeError(
                            f"work_type classification failed for '{title}': {labels_result.get('rationale')}"
                        )
                    work_type_labels = _normalize_work_type_labels(labels_result.get("labels") or [])

                    if _is_news_report_title(title) and "news/report" not in [label.lower() for label in work_type_labels]:
                        work_type_labels.append("News/Report")

                record = build_record(
                    main_task=main_task,
                    work=work,
                    paper_alias=alias,
                    paper_url=paper_url,
                    work_type_labels=work_type_labels,
                    bibtex=bibtex_result.bibtex,
                    bibtex_status=bibtex_result.status,
                    method_or_bench_name=method_name,
                    source_used="|".join(record_sources),
                    parent_seed=root_seed_by_node.get(node_key, ""),
                )
                record["multi_turn_signal"] = _annotate_multi_turn_signal(multi_turn_signal, kept=True)
                record["relevance_rationale"] = relevance_rationale
                records.append(record)
                _write_live_row(records_fh, records_writer, record, CSV_COLUMNS)
                _log(args.verbose, f"[OK] {len(records):04d} {title}")
                print(
                    f"[PROGRESS] processed={len(records) + len(dropped_records)} "
                    f"kept={len(records)} dropped={len(dropped_records)} queue={len(queue)}"
                )

            root_seed = root_seed_by_node.get(node_key, "")
            if (not args.disable_openalex) and node.openalex_id and node.edge_source in {"openalex_exact", "openalex_edge"} and work:
                backward_ids = get_backward_reference_ids(work, max_items=args.max_related_per_paper)
                forward_ids = get_forward_citation_ids(node.openalex_id, max_items=args.max_related_per_paper)
                for candidate in backward_ids + forward_ids:
                    if not candidate:
                        continue
                    candidate_id = _as_openalex_id({"id": candidate})
                    if not candidate_id:
                        continue
                    fetched = get_work_by_openalex_id(candidate_id)
                    candidate_title = str((fetched or {}).get("display_name") or "")
                    candidate_url = choose_preferred_paper_url(fetched) if fetched else ""
                    enqueue_node(
                        title=candidate_title,
                        openalex_id=candidate_id,
                        paper_url=candidate_url,
                        edge_source="openalex_edge",
                        parent_seed=root_seed,
                        apply_coarse_filter=True,
                    )
            else:
                backward_candidates = get_backward_reference_candidates_from_pdf(
                    current_paper_url,
                    max_items=args.max_related_per_paper,
                )
                forward_candidates = get_forward_citation_candidates(
                    title,
                    provider_priority=provider_priority,
                    max_items=args.max_related_per_paper,
                )
                for candidate in backward_candidates + forward_candidates:
                    candidate_title = str(candidate.get("title") or "").strip()
                    candidate_url = _normalize_seed_url(str(candidate.get("paper_url") or ""))
                    candidate_source = str(candidate.get("source") or "Fallback")
                    candidate_openalex = ""
                    if (not args.disable_openalex) and args.prefer_openalex_exact and candidate_title:
                        exact = resolve_title_to_openalex_exact(
                            candidate_title,
                            title_threshold=args.openalex_title_exact_threshold,
                        )
                        if exact:
                            candidate_openalex = _as_openalex_id(exact)
                            candidate_title = str(exact.get("display_name") or candidate_title)
                            if not candidate_url:
                                candidate_url = choose_preferred_paper_url(exact)
                            candidate_source = "openalex_exact"
                    enqueue_node(
                        title=candidate_title,
                        openalex_id=candidate_openalex,
                        paper_url=candidate_url,
                        edge_source=candidate_source,
                        parent_seed=root_seed,
                        apply_coarse_filter=True,
                    )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        records_fh.close()
        dropped_fh.close()

    stats_path = write_work_type_stats_csv(records, args.stats_output)

    print(f"[DONE] Mode: {args.mode}")
    print(f"[DONE] Exported {len(records)} records")
    print(f"[DONE] Dropped {len(dropped_records)} records")
    print(f"[DONE] CSV: {output_path}")
    print(f"[DONE] Stats: {stats_path}")
    print(f"[DONE] Dropped CSV: {dropped_path}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
