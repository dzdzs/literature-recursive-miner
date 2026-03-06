# literature-recursive-miner

Recursively discover literature and stream results to CSV in real time:
- Input: topic/keywords, seed papers, runtime classification rules
- Expansion: prioritize PDF references + Scholar/SerpAPI/Semantic Scholar citations
- Output: `papers.csv` / `dropped.csv` / `stats.csv`

## 0. Pipeline Overview

The workflow is queue-based recursion with incremental writes:

1. Load runtime arguments and `.env`
   - Read topic, keywords, filter conditions, work_type rules, seeds, and limits.
2. Enqueue seeds
   - Normalize each seed into a canonical paper URL (prefer arXiv `https://arxiv.org/pdf/<id>`).
   - Build the processing queue.
3. Process papers one by one (loop)
   - Pop one paper and fetch metadata/abstract first.
   - If abstract is missing: download PDF and run `pdftotext` to extract abstract/snippet as classification evidence.
   - Query DBLP for the latest BibTeX (`missing_dblp` if unavailable).
4. Relevance decision (LLM)
   - Decide keep/drop using `THEME/KEYWORD/FILTER_* / RELEVANCE_INSTRUCTION`.
   - Irrelevant papers go to `dropped.csv`; relevant papers go to `papers.csv`.
5. Work type classification (LLM)
   - Label with `Bench / Agent Evo / Env Evo / Survey / News/Report`, etc., based on `WORK_TYPE_INSTRUCTION`.
6. Recursively expand next-layer candidates
   - Backward: extract candidate titles from the current paper's PDF references.
   - Forward: fetch citing candidates from Scholar/SerpAPI/Semantic Scholar.
   - Run coarse filtering, enqueue candidates, and continue.
7. Real-time CSV write + final stats
   - Append to `papers.csv` or `dropped.csv` immediately after each processed paper.
   - Output `stats.csv` at the end (counts by work_type).

## 1. Dependencies

- `python3`
- `pdftotext` (for PDF -> text)
- Network access
- LLM API key (`OPENAI_API_KEY` or `API_KEY`)

## 2. Configure `.env`

A `.env` template is provided at the project root. Update key fields as needed:

- `OPENAI_API_KEY` or `API_KEY` (one is required)
- `API_BASE_URL`
- `OPENAI_MODEL`
- `TOPIC` / `THEME` / `KEYWORD`
- `SEED_1..SEED_3`
- `WORK_TYPE_INSTRUCTION`

## 3. Run

```bash
cd /mnt/userdata/huashengjia/literature-recursive-miner
./run_lrm.sh /mnt/userdata/huashengjia/lrm_run_001
```

During execution, progress is printed in real time:

- `[PROGRESS] processed=... kept=... dropped=... queue=...`

After completion, outputs are written to your target directory:

- `papers.csv`
- `dropped.csv`
- `stats.csv`

## 4. `.env` Loading Rule

`run_lrm.sh` loads `.env` from the **current working directory**, i.e., `$(pwd)/.env`.

Example: if you run the script from another directory, it reads the `.env` in that directory:

```bash
cd /some/dir/with/env
/mnt/userdata/huashengjia/literature-recursive-miner/run_lrm.sh /tmp/lrm_out
```

## 5. Troubleshooting

- Error `Missing API key`:
  - Check whether `OPENAI_API_KEY` or `API_KEY` is set in `.env`.
- Too few results:
  - Increase `MAX_PAPERS` and `MAX_RELATED_PER_PAPER`;
  - Relax `FILTER_*` or `RELEVANCE_INSTRUCTION`.

## 6. Chinese Versions

- Chinese README: `README_zh.md`
- Chinese launcher script: `run_lrm_zh.sh`
