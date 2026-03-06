# literature-recursive-miner

用于递归挖掘文献并实时写出结果到 CSV：
- 输入：主题/关键词、种子论文、运行时分类规则
- 扩展：优先 PDF 参考文献 + Scholar/SerpAPI/Semantic Scholar 被引
- 输出：`papers.csv` / `dropped.csv` / `stats.csv`

## 0. 项目流程

整体是一个“队列递归 + 实时落盘”的流程：

1. 读取运行参数与 `.env`
   - 读取主题、关键词、过滤条件、work_type 规则、seed 列表、最大篇数等。
2. seed 入队
   - 每个 seed 先转成统一论文 URL（优先 arXiv `https://arxiv.org/pdf/<id>`）。
   - 建立待处理队列（queue）。
3. 逐篇处理（循环）
   - 取出 1 篇论文，优先拿元数据与摘要。
   - 若摘要缺失：自动下载 PDF，用 `pdftotext` 抽取 abstract/snippet 作为分类证据。
   - 查 DBLP 获取最新 BibTeX（找不到标记 `missing_dblp`）。
4. 相关性判定（LLM）
   - 根据 `THEME/KEYWORD/FILTER_* / RELEVANCE_INSTRUCTION` 判断是否保留。
   - 不相关写入 `dropped.csv`；相关写入 `papers.csv`。
5. 工作类型分类（LLM）
   - 按 `WORK_TYPE_INSTRUCTION` 打 `Bench / Agent Evo / Env Evo / Survey / News/Report` 等标签。
6. 递归扩展下一层候选
   - 向后：从当前论文 PDF 的 References 提取候选标题。
   - 向前：从 Scholar/SerpAPI/Semantic Scholar 拉取被引候选。
   - 候选经粗过滤后入队，继续循环。
7. 实时写 CSV + 结束统计
   - 每处理一篇就立刻追加到 `papers.csv` 或 `dropped.csv`（不是最后一次性写）。
   - 结束后输出 `stats.csv`（按 work_type 计数）。

## 1. 依赖

- `python3`
- `pdftotext`（用于 PDF -> 文本）
- 可访问网络
- LLM API Key（`OPENAI_API_KEY` 或 `API_KEY`）

## 2. 配置 .env

项目根目录已提供 `.env`，按需修改关键字段：

- `OPENAI_API_KEY` 或 `API_KEY`（必填其一）
- `API_BASE_URL`
- `OPENAI_MODEL`
- `TOPIC` / `THEME` / `KEYWORD`
- `SEED_1..SEED_3`
- `WORK_TYPE_INSTRUCTION`

## 3. 运行方式

```bash
cd /mnt/userdata/huashengjia/literature-recursive-miner
./run_lrm.sh /mnt/userdata/huashengjia/lrm_run_001
```

运行时会实时打印：

- `[PROGRESS] processed=... kept=... dropped=... queue=...`

完成后输出在你指定目录：

- `papers.csv`
- `dropped.csv`
- `stats.csv`

## 4. .env 加载规则（已改）

`run_lrm.sh` 会从**当前执行目录**加载 `.env`，即 `$(pwd)/.env`。

示例：如果你在其他目录运行脚本，脚本会读取那个目录下的 `.env`：

```bash
cd /some/dir/with/env
/mnt/userdata/huashengjia/literature-recursive-miner/run_lrm.sh /tmp/lrm_out
```

## 5. 常见问题

- 报错 `Missing API key`：
  - 检查 `.env` 里的 `OPENAI_API_KEY`/`API_KEY` 是否已填写。
- 结果太少：
  - 增大 `MAX_PAPERS`、`MAX_RELATED_PER_PAPER`；
  - 放宽 `FILTER_*` 或 `RELEVANCE_INSTRUCTION`。
