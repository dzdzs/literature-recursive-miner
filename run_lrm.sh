#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/scripts/run_pipeline.py"

OUT_DIR="${1:-${SCRIPT_DIR}/run_output}"
mkdir -p "${OUT_DIR}"

# Load env from current execution directory (PWD), not OUT_DIR.
ENV_FILE="${PWD}/.env"

load_env_file() {
  local env_path="$1"
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line:0:1}" == "#" ]] && continue

    if [[ "${line}" == export[[:space:]]* ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "${line}" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"

    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if [[ "${value}" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done < "${env_path}"
}

if [[ -f "${ENV_FILE}" ]]; then
  load_env_file "${ENV_FILE}"
  echo "[INFO] loaded env file: ${ENV_FILE}"
else
  echo "[WARN] env file not found at ${ENV_FILE}; using existing shell env only"
fi

MAX_PAPERS="${MAX_PAPERS:-300}"
MAX_PROCESSED="${MAX_PROCESSED:-0}"
MAX_RELATED_PER_PAPER="${MAX_RELATED_PER_PAPER:-100}"

API_KEY_VALUE="${OPENAI_API_KEY:-${API_KEY:-}}"
API_BASE_URL_VALUE="${API_BASE_URL:-https://api.gpt.ge}"
OPENAI_MODEL_VALUE="${OPENAI_MODEL:-gpt-5.2}"
CITATION_PROVIDER_PRIORITY_VALUE="${CITATION_PROVIDER_PRIORITY:-scholar_html,serpapi,semantic_scholar}"
OPENALEX_TITLE_EXACT_THRESHOLD_VALUE="${OPENALEX_TITLE_EXACT_THRESHOLD:-0.95}"
DISABLE_OPENALEX_VALUE="${DISABLE_OPENALEX:-1}"

TOPIC_VALUE="${TOPIC:-deep research}"
THEME_VALUE="${THEME:-deep research}"
KEYWORD_VALUE="${KEYWORD:-search}"
FILTER_1_VALUE="${FILTER_1:-主题为deep research或search相关}"
FILTER_2_VALUE="${FILTER_2:-多轮优先但非硬过滤，非多轮也保留并在multi_turn_signal标注}"
FILTER_3_VALUE="${FILTER_3:-必须是agent相关工作；若论文主要是RAG，只有明确为agentic RAG（服务于agent的规划/工具使用/多步决策/行动闭环）才保留。}"
RELEVANCE_INSTRUCTION_VALUE="${RELEVANCE_INSTRUCTION:-优先保留与deep research/search及agent相关的条目。若是RAG，优先保留agentic RAG；普通RAG可在证据不足时标记uncertain_keep后保留。对于GAIA这类Agent benchmark，若是否涉及搜索能力不确定，也先保留并在rationale中注明uncertain_keep。产品公告/博客/报告（如Deep Research功能可用性公告）只要主题相关也保留。multi-turn仅作为标注，不是硬性过滤条件。}"
SEED_1_VALUE="${SEED_1:-https://arxiv.org/pdf/2506.11763}"
SEED_2_VALUE="${SEED_2:-https://arxiv.org/pdf/2504.03160}"
SEED_3_VALUE="${SEED_3:-https://arxiv.org/pdf/2311.12983}"
WORK_TYPE_INSTRUCTION_VALUE="${WORK_TYPE_INSTRUCTION:-Bench：环境的设计以及在其基础上的评测分析；Agent Evo：使用固定环境对agent训练的工作，包括agent workflow，能够提升agent能力的都算，但是太简单太基本的不要；Env Evo：环境和Agent协同进化的工作；Survey：综述/系统性综述/综述性回顾类论文，单独归类为Survey；News/Report：产品公告、博客、功能发布、行业报告等非论文但与主题相关的条目。}"

if [[ ! -f "${PIPELINE}" ]]; then
  echo "[ERROR] pipeline not found: ${PIPELINE}" >&2
  exit 1
fi

if [[ -z "${API_KEY_VALUE}" ]]; then
  echo "[ERROR] Missing API key. Set OPENAI_API_KEY or API_KEY." >&2
  exit 1
fi

echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] model: ${OPENAI_MODEL_VALUE}"
echo "[INFO] api base: ${API_BASE_URL_VALUE}"

python3 "${PIPELINE}" \
  --mode assistant-run \
  --api-key "${API_KEY_VALUE}" \
  --api-base-url "${API_BASE_URL_VALUE}" \
  --llm-model "${OPENAI_MODEL_VALUE}" \
  --topic "${TOPIC_VALUE}" \
  --theme "${THEME_VALUE}" \
  --keyword "${KEYWORD_VALUE}" \
  --filter-condition "${FILTER_1_VALUE}" \
  --filter-condition "${FILTER_2_VALUE}" \
  --filter-condition "${FILTER_3_VALUE}" \
  --relevance-instruction "${RELEVANCE_INSTRUCTION_VALUE}" \
  --seed "${SEED_1_VALUE}" \
  --seed "${SEED_2_VALUE}" \
  --seed "${SEED_3_VALUE}" \
  --work-type-instruction "${WORK_TYPE_INSTRUCTION_VALUE}" \
  --max-papers "${MAX_PAPERS}" \
  --max-processed "${MAX_PROCESSED}" \
  --max-related-per-paper "${MAX_RELATED_PER_PAPER}" \
  --citation-provider-priority "${CITATION_PROVIDER_PRIORITY_VALUE}" \
  --openalex-title-exact-threshold "${OPENALEX_TITLE_EXACT_THRESHOLD_VALUE}" \
  --topic-keyword "deep research" \
  --topic-keyword "search" \
  --topic-keyword "research" \
  --topic-keyword "retrieval" \
  --topic-keyword "information seeking" \
  --topic-keyword "question answering" \
  --agent-keyword "agent" \
  --agent-keyword "agentic" \
  --agent-keyword "assistant" \
  --agent-keyword "autonomous" \
  --agent-keyword "multi-agent" \
  --agent-keyword "workflow" \
  --agentic-rag-keyword "agentic rag" \
  --agentic-rag-keyword "tool use" \
  --agentic-rag-keyword "planning" \
  --agentic-rag-keyword "workflow" \
  --agentic-rag-keyword "action loop" \
  --agentic-rag-keyword "orchestration" \
  $( [[ "${DISABLE_OPENALEX_VALUE}" == "1" ]] && printf '%s' '--disable-openalex' ) \
  --sleep-seconds 0 \
  --output "${OUT_DIR}/papers.csv" \
  --dropped-output "${OUT_DIR}/dropped.csv" \
  --stats-output "${OUT_DIR}/stats.csv"

echo "[DONE] papers:  ${OUT_DIR}/papers.csv"
echo "[DONE] dropped: ${OUT_DIR}/dropped.csv"
echo "[DONE] stats:   ${OUT_DIR}/stats.csv"
