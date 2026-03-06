# Runtime work_type Contract

## Goal

`work_type` must be configurable at runtime. Do not hardcode one fixed taxonomy in code.
Default execution mode is `assistant-run` (no local LLM call). Local LLM classification is optional.

## Runtime inputs

Every run must include:
- `theme` (relevance theme)
- optional `filter_condition` list (for extra constraints like multi-turn environments)
- `work_type_instruction` in natural language (required in `local-llm` mode).

Examples:

- "Classify as Bench / Agent Evo / Env Evo. Allow multi-label."
- "Use labels Scenario Design / Policy Learning / Co-Evolution."

## Classification policy

- In `assistant-run` mode: assistant judges relevance/work_type in chat; script exports candidates.
- In `local-llm` mode: use LLM semantic judgment; do not rely on keyword-only matching.
- Read title + abstract first; use full-text snippets when title/abstract is insufficient.
- Return zero, one, or multiple labels according to instruction.

## Output policy

- Save labels in `work_type` as `|`-separated values.
- In `assistant-run`, labels can remain empty until assistant-side review finishes.
- In `local-llm`, fail fast if `OPENAI_API_KEY` is missing or LLM classification fails.
