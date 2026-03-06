#!/usr/bin/env python3
"""CSV export helpers for literature-recursive-miner."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "main_task",
    "paper_alias",
    "paper_title",
    "paper_url",
    "work_type",
    "first_author_affiliation",
    "publish_time",
    "bibtex",
    "bibtex_status",
    "method_or_bench_name",
    "source_used",
    "parent_seed",
    "multi_turn_signal",
    "relevance_rationale",
]

DROPPED_COLUMNS = [
    "main_task",
    "theme",
    "paper_alias",
    "paper_title",
    "paper_url",
    "parent_seed",
    "drop_reason",
    "multi_turn_signal",
    "relevance_rationale",
]


def write_records_csv(records: list[dict[str, Any]], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    return path


def write_work_type_stats_csv(records: list[dict[str, Any]], output_path: str) -> Path:
    counter: Counter[str] = Counter()
    for row in records:
        raw = str(row.get("work_type", "")).strip()
        if not raw:
            counter["Unclassified"] += 1
            continue
        for label in [item.strip() for item in raw.split("|") if item.strip()]:
            counter[label] += 1

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["work_type", "paper_count"])
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([label, count])
    return path


def write_dropped_csv(records: list[dict[str, Any]], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DROPPED_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow({col: row.get(col, "") for col in DROPPED_COLUMNS})
    return path
