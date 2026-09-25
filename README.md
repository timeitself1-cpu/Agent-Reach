# Agent Reach v2

Multi-source trend intelligence: async ingestion -> heuristic noise filter -> page-content enrichment -> embedding + HDBSCAN density clustering -> `llama3.1:8b` labelling -> relevance + momentum scoring -> SQLite (WAL) history -> ASCII executive report with a balanced item ledger.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull llama3.1:8b                                 # Ollama must be running on localhost:11434
ollama pull nomic-embed-text                            # embeddings for density clustering (274 MB)
copy .env.example .env                                  # optional; set AGENT_REACH_CONTACT_EMAIL
```

## Run

```bash
python -m agent_reach                                   # one run
python -m agent_reach --loop --interval 30              # every 30 min (builds the velocity history)
python -m agent_reach --sources hackernews github arxiv --top 10
python -m agent_reach --no-llm                          # deterministic clustering, no Ollama
python -m agent_reach --json-out reports/latest.json --log-level DEBUG
```

## Chat with Reach

```bash
python -m agent_reach chat                  # opens http://127.0.0.1:8765 in your browser
python -m agent_reach chat --no-cloud       # local runs only
python -m agent_reach chat --model llama3.1:8b --port 8800
```

A local web page: the latest trends on the left, a chat with **Reach** on the right.

- **Which report:** the newer of your latest local run (SQLite history) and the newest successful `cloud-runner` run on `main`. The cloud report is downloaded once with the GitHub CLI (`gh`) and cached under `reports/cloud/`. **Refresh** looks again.
- **Grounded:** Reach answers only from the report (trends, scores, summaries, links, source health) and the headlines of earlier local runs. It says so when the report doesn't cover a question.
- **Deep reads:** when a question names a trend (`#2`, `trend 3`, or words from its headline), Reach first reads that trend's source pages with the pipeline's page extractor, so it can go beyond the summary.
- **Local only:** replies stream from your Ollama (`AGENT_REACH_CHAT_MODEL`, else `AGENT_REACH_OLLAMA_MODEL`). The server listens on 127.0.0.1 only and rejects other Host headers and non-JSON posts.

## Architecture

| Module | Role |
|---|---|
| `config.py` | Pydantic Settings; every field overridable via `AGENT_REACH_*` env vars / `.env` |
| `models.py` | `CategoryEnum`, `RawTrendItem`, `CleanedTrendItem` (+ scraped `context`), `MacroCluster`, `PipelineAccounting` (ledger with invariant checks), `PipelineReport` (schema_version 2) |
| `ingestion/base.py` | `BaseIngester`: shared `httpx.AsyncClient`, per-call timeout, exponential backoff + jitter, `Retry-After`, extra retryable statuses, per-ingester request pacing; `run()` never raises |
| `ingestion/social.py` | X (Trends24 scrape), Reddit (JSON with score/comments -> RSS; 5 s timeout, 403/429 retries, 2 s pacing, 30 s budget), TikTok Creative Center (5 s timeout, retries, 2 s pacing) |
| `ingestion/search.py` | Google Trends RSS, Google News RSS, Wikipedia top pageviews, ArXiv Atom API |
| `ingestion/papers.py` | DAIR.AI "AI Papers of the Week" (newest week from `years/<year>.md` on GitHub) |
| `ingestion/tech.py` | Hacker News (Algolia), GitHub Trending scrape, Product Hunt feed |
| `pipeline/cleaner.py` | ASCII normalisation, hashtag splitting, engagement thresholds, noise regexes, de-dup, heuristic score, output sanitisers |
| `pipeline/enricher.py` | Stage 2b: fetches each candidate's page (or Wikipedia summary API) and keeps title + meta description + 1-2 lead paragraphs (trafilatura, BeautifulSoup fallback) |
| `pipeline/density.py` | Stage 3a: `nomic-embed-text` embeddings -> HDBSCAN (leaf selection) -> cosine-to-centroid gate; outliers are noise |
| `pipeline/clusterer.py` | Stage 3b-e: LLM labelling only (never grouping), `[INSUFFICIENT_DATA]` flag, entity isolation + orphan re-homing, merge, drop rules, guardrails |
| `pipeline/scorer.py` | Relevance 1-10 (LLM + heuristics), velocity 0-100 vs 1h/6h/24h snapshots |
| `storage/db.py` | SQLite WAL: `runs`, `raw_items`, `clusters`, `entity_snapshots` |
| `main.py` | Stage orchestration (1 ingest, 2 clean/select, 2b enrich, 3 cluster, 4 score, 5 persist), accounting ledger, report renderer, loop mode |

## Pipeline stages

1. **Ingest.** Eleven sources are fetched concurrently. A failing source returns no items; it never fails the run.
2. **Clean and select.** Engagement thresholds, noise regexes and de-duplication run first. Then the top `MAX_ITEMS_FOR_LLM` items, with a floor per source, become clustering candidates.
3. **Enrich (2b).** Each candidate gets `context`: page title, meta description and 1-2 lead paragraphs.
   - Wikipedia uses its summary API.
   - arXiv, AI Papers of the Week and Product Hunt use their feed text.
   - X and TikTok have no article page, so their items get no context.
4. **Cluster (3a-3e).**
   - **3a density:** embed `title + context`, run HDBSCAN (`min_cluster_size=2`, leaf selection), then apply a cosine-to-centroid gate. Outliers are noise and are dropped. They are never forced into a mixed bucket.
   - **3b label:** the LLM only names groups (headline, category, entities, two sentences, relevance). Groups whose signals can't explain what happened and why get `[INSUFFICIENT_DATA]`.
   - **3c entity isolation:** a group must be connected by distinctive tokens, a shared named entity, or entities that co-occur elsewhere in the run. Mixed groups are split and re-labelled. An orphan re-joins a cluster only when it literally names that cluster's grounded entity. The LLM may propose a home for an orphan, but the proposal is kept only when the orphan shares distinctive tokens with a member or names a grounded entity.
   - **3d merge:** groups that resolve to identical entities are merged.
   - **3e drop:** clusters flagged `[INSUFFICIENT_DATA]`, with filler summaries ("no specific information", "details are scarce"), or with relevance <= 3 are dropped, as are weak singletons.
5. **Score and persist.** Relevance and velocity are computed, then everything is saved to SQLite.

With no embedding model, grouping falls back to lexical union-find under the same noise rule. With no Ollama, labels are heuristic, and summaries use the scraped lead sentence or are flagged insufficient.

## Item accounting

Every run prints and stores a raw-item ledger (`PipelineReport.accounting`). The invariant is:

    ingested == sum(discarded[reason]) + clustered        (clustered == sum(raw_item_count) over clusters)

- **filtered:** noise-filter reasons, such as `reddit_low_engagement`, `generic_hashtag` or `pet_post`.
- **budget:** `not_selected_budget` counts items that passed filters but ranked below the clustering cap.
- **clustering:** `density_noise`, `unsupported_grouping`, `insufficient_data`, `low_relevance` and `weak_singleton`.

A de-duplicated item stands for all its duplicates (`raw_weight`), so de-duplication is never a discard. The ledger validates stage by stage, and an imbalance is logged as an error.

## Noise defences (pre-LLM)

- **Source thresholds:** Reddit score < 20 or comments < 5, HN points < 10, GitHub stars-today < 20, and low-signal subreddits are all dropped. Reddit RSS items are kept only when they rank in the top `REDDIT_RSS_MAX_RANK` of a subreddit's top-of-day listing.
- **Regex rules:** clickbait prefixes are stripped. Personal anecdotes, meme/photo and pet posts, betting/box-score chatter and generic hashtags are dropped.

## Reddit and TikTok resilience

- **Reddit:** every request uses fixed browser headers, a 5 s timeout and exponential-backoff retries on 403/429/5xx/timeouts, honouring `Retry-After`. Requests are paced at least 2 s apart. After the first definitive JSON 403/429, the remaining subreddits use RSS, and a 30 s budget stops new subreddits from starting.
- **TikTok:** 5 s timeout, 2 retries on 403/429/5xx, paced 2 s apart. It is often bot-gated and then reports `FAIL` cleanly.

## Velocity

For each entity: `rate = % of kept items mentioning it`. For each lookback window, the completed run closest to `now - w` (within ±50%) is found, and the rate is recomputed from that run's stored titles. `growth = (now - then) / max(then, one_item_floor)`. `velocity = 50 + 50 * weighted_mean(tanh(growth/2))`, with weights 0.5, 0.3 and 0.2 for 1h, 6h and 24h. 50 means flat. Momentum labels: NEW, SURGING (75+), RISING (60+), STEADY, COOLING (25+), FADING. Until history exists, the score is a cold-start estimate labelled `BASELINE`. Run in `--loop` mode to fill the 1h, 6h and 24h windows.

## Notes

- Titles are folded to ASCII (as specified), so non-Latin-script trends are discarded. For another region, change `AGENT_REACH_GEO`, `TRENDS24_REGION` and `WIKIPEDIA_PROJECT`.
- Trends24, GitHub Trending and TikTok are HTML scrapes. When their markup changes, the ingester reports `FAIL` in SOURCE HEALTH and the run continues.
- TikTok Creative Center often bot-gates unauthenticated requests. Expect intermittent `FAIL` from that source.
- Unreachable Ollama, a missing model or invalid JSON all fall back to deterministic grouping and heuristic labels. The engine and the grouping method are shown in the report header.
- `density_member_min_cosine` (0.55) and `hdbscan_selection` (`leaf`) are tuned for `nomic-embed-text`. If you change the embedding model, re-check the `stage 3a density` log line.

## Cloud runner (GitHub Actions)

`run_cloud_handoff.ps1` syncs this folder to `timeitself1-cpu/Agent-Reach`. It excludes secrets, blocks token-looking strings, and keeps remote-only files unless you pass `-Mirror`. It then dispatches `.github/workflows/cloud-runner.yml`, waits until the run is `in_progress`, and streams its logs.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cloud_handoff.ps1                 # full LLM run
.\run_cloud_handoff.ps1 -NoLLM -ExtraArgs "--top 10"                              # fast deterministic run
```

The workflow installs Ollama on a CPU-only `ubuntu-latest` runner and caches `llama3.1:8b` after the first pull. It also pulls `nomic-embed-text`. It clusters up to 150 candidates, and because grouping is embedding-based, the LLM only labels real clusters. SQLite history is carried between runs in the Actions cache, so velocity works in the cloud as well. The report appears in the run summary and as a downloadable artifact. Optionally, set the repository variable `AGENT_REACH_CONTACT_EMAIL`.
