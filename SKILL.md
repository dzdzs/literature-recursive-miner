---
name: literature-recursive-miner
description: Recursive literature discovery and structuring for a topic. Use when the user provides topic/keywords plus seed papers and needs backward+forward citation expansion, runtime user-defined relevance filtering conditions (theme and extra constraints such as multi-turn), CSV output, runtime-definable work_type classification, and strict DBLP-only BibTeX retrieval (no paper-copy and no fabricated citation entries). Default mode is assistant-run with real-time LLM filtering and classification.
---

# Literature Recursive Miner

## Overview

Run a recursive paper-mining workflow from seed papers, expand by references and citations, then export structured CSV for analysis.
Keep `work_type` classification runtime-configurable through a natural-language instruction, and treat BibTeX as DBLP-only.

## Workflow

1. Parse user input
- Collect `topic`, `theme`, `keywords`, `filter_condition` (optional, repeatable), `seed papers`, `work_type_instruction`, and optional limits.
- Default to `max_papers=300` kept papers unless user overrides it.
- Optional `max_processed` can cap total processed papers (`kept + dropped`) for runtime control.

2. Resolve seeds and expand graph
- Prefer exact OpenAlex match for each seed (`DOI/arXiv/OpenAlex ID/title exact-threshold`).
- If exact OpenAlex match exists for a paper, expand with OpenAlex edges:
  - backward references
  - forward citations
- If exact OpenAlex match is unavailable, fallback expansion:
  - backward from PDF `References/Bibliography` parsing (`pdftotext`)
  - forward from provider priority: `Google Scholar HTML -> SerpAPI Scholar -> Semantic Scholar`
- Apply queue coarse filter before enqueueing candidates (topic + agent + agentic-RAG keywords).
- Continue until no new papers appear or max limit is reached.

3. Choose run mode
- `assistant-run` (default): run real-time relevance/work_type classification using API credentials.
- `local-llm` (compat mode): same behavior as `assistant-run`.
- `collect-only` (optional): do not call LLM; export candidate set only.

4. Runtime relevance filtering (assistant-run/local-llm)
- Evaluate each candidate with LLM relevance judgment using user-provided `theme` and `filter_condition`.
- Keep only relevant papers in `papers.csv`.
- Write filtered-out papers to `dropped.csv` with drop reason and rationale.
- Do not expand references/citations from dropped papers.
- `multi_turn_signal` is annotation-first: non-multi-turn papers can still be kept if otherwise relevant.

5. Normalize required fields
- Fill required fields for each paper:
  - `main_task`
  - `paper_alias` (prefer extracted method/benchmark name, e.g. `MAKG`)
  - `paper_url`
  - `work_type`
  - `first_author_affiliation`
  - `publish_time`
  - `bibtex`
  - `method_or_bench_name`
- Prefer arXiv HTTPS PDF URL first, OpenReview second, other stable URLs last.
- Prefer arXiv ID month/year for `publish_time`; otherwise use venue publication date.

6. Enforce strict BibTeX policy
- Search DBLP by paper title.
- Copy BibTeX from DBLP record only.
- If DBLP has no valid match, mark `bibtex_status=missing_dblp` and leave `bibtex` empty.
- Never extract BibTeX from paper text and never fabricate entries.

7. Runtime `work_type` classification
- Use user-provided natural-language `work_type_instruction` every run.
- In `assistant-run`/`local-llm` mode, let LLM judge labels based on title/abstract/full-text snippets when needed.
- Support multi-label output.

8. Export outputs
- Write main paper table as CSV (`papers.csv`).
- Write dropped paper table as CSV (`dropped.csv`).
- Write a second CSV with preliminary `work_type` counts.
- Append rows to `papers.csv`/`dropped.csv` in real time during execution (not only at the end).

## Scripts

- `scripts/run_pipeline.py`
  - End-to-end pipeline runner.
- `scripts/collect_sources.py`
  - OpenAlex retrieval, URL selection, citation expansion, DBLP BibTeX lookup.
- `scripts/extract_fields.py`
  - Publish time inference, method/bench extraction, alias/record building.
- `scripts/classify_paper.py`
  - LLM-based runtime classifier using user instruction.
- `scripts/export_csv.py`
  - CSV export for records and statistics.

## Run

```bash
python3 scripts/run_pipeline.py \
  --mode "assistant-run" \
  --api-key "$OPENAI_API_KEY" \
  --api-base-url "${API_BASE_URL:-https://api.openai.com/v1}" \
  --llm-model "${OPENAI_MODEL:-gpt-4.1-mini}" \
  --topic "Embodied agent training" \
  --theme "deep research" \
  --keyword "benchmark" \
  --filter-condition "environment should be multi-turn" \
  --citation-provider-priority "scholar_html,serpapi,semantic_scholar" \
  --openalex-title-exact-threshold 0.95 \
  --topic-keyword "deep research" \
  --agent-keyword "agent" \
  --agentic-rag-keyword "agentic rag" \
  --seed "Voyager: An Open-Ended Embodied Agent with Large Language Models" \
  --seed "https://openreview.net/forum?id=..." \
  --work-type-instruction "Label as Bench / Agent Evo / Env Evo; allow multi-label if clearly justified." \
  --max-papers 300 \
  --output papers.csv \
  --dropped-output dropped.csv \
  --stats-output work_type_stats.csv
```

Collect-only example:

```bash
python3 scripts/run_pipeline.py \
  --mode "collect-only" \
  --topic "Embodied agent training" \
  --theme "deep research" \
  --keyword "benchmark" \
  --filter-condition "environment should be multi-turn" \
  --seed "Voyager: An Open-Ended Embodied Agent with Large Language Models" \
  --output papers.csv \
  --dropped-output dropped.csv \
  --stats-output work_type_stats.csv
```

## References

Load only as needed:
- `references/source-priority.md` for strict source ordering and fallback behavior.
- `references/classification-contract.md` for runtime work_type instruction contract.
