# Agent Reach v2

Multi-source trend intelligence: async ingestion -> heuristic noise filter -> `llama3.1:8b` semantic clustering -> relevance + momentum scoring -> SQLite (WAL) history -> ASCII executive report.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull llama3.1:8b                                 # Ollama must be running on localhost:11434
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

## Architecture

| Module | Role |
|---|---|
| `config.py` | Pydantic Settings; every field overridable via `AGENT_REACH_*` env vars / `.env` |
| `models.py` | `CategoryEnum`, `RawTrendItem`, `CleanedTrendItem`, `MacroCluster`, `PipelineReport`, LLM I/O schemas |
| `ingestion/base.py` | `BaseIngester`: shared `httpx.AsyncClient`, 10s timeout, exponential backoff + jitter, `Retry-After`, UA rotation; `run()` never raises |
| `ingestion/social.py` | X (Trends24 scrape), Reddit (JSON with score/comments, RSS fallback), TikTok Creative Center (`__NEXT_DATA__`) |
| `ingestion/search.py` | Google Trends RSS, Google News RSS, Wikipedia top pageviews, ArXiv Atom API |
| `ingestion/tech.py` | Hacker News (Algolia), GitHub Trending scrape, Product Hunt feed |
| `pipeline/cleaner.py` | ASCII normalisation, hashtag splitting, engagement thresholds, noise regexes, de-dup, heuristic score |
| `pipeline/clusterer.py` | Lexical pre-grouping -> batched JSON-schema Ollama calls -> merge pass -> category guardrails |
| `pipeline/scorer.py` | Relevance 1-10 (LLM + heuristics), velocity 0-100 vs 1h/6h/24h snapshots |
| `storage/db.py` | SQLite WAL: `runs`, `raw_items`, `clusters`, `entity_snapshots` |
| `main.py` | `asyncio.gather` orchestration, persistence, report renderer, loop mode |

## Noise defences

1. **Source thresholds:** Reddit score < 20 or comments < 5, HN points < 10, GitHub stars-today < 20, and low-signal subreddits (r/aww, r/pics, r/AskReddit, and similar) are all dropped. Reddit RSS has no metrics, so RSS items are kept only when they rank in the top `REDDIT_RSS_MAX_RANK` (default 10) of a subreddit's top-of-day listing.
2. **Regex rules:** clickbait prefixes are stripped. Personal anecdotes, meme/photo posts, pet posts, betting/box-score chatter and generic hashtags (`#fyp`, `#fallvibes`, `#MondayMotivation`) are dropped.
3. **LLM discards:** the model is told to discard leftover noise.
4. **Coherence enforcement:** every LLM cluster must form one connected graph. Two items connect when they share distinctive tokens, or when both literally name the same entity. A mixed ("Frankenstein") cluster is split into its cores, and each core is re-titled by a second LLM pass that labels groups without regrouping them. Stray items are re-assigned or dropped. Merge-pass proposals are checked with the same test, and unsupported ones are rejected.
5. **Singleton policy:** a one-item, one-source cluster survives only when its heuristic score is at least 0.80 or its LLM relevance is at least 7.
6. **Category guardrails:** a cluster labelled Tech or Science & AI with no tech evidence (no tech source and no tech keywords) is reassigned by keyword vote. Clusters made only of ArXiv items are always Science & AI.
7. **Output sanitisation:** headlines become Title Case, max 10 words, with generic umbrella titles replaced by the best member title. Summaries have URLs, @-markers and JSON leakage stripped, then are cut to exactly two capitalised sentences.

## LLM batching

`llama3.1:8b` gets 20 items per call (`AGENT_REACH_LLM_BATCH_SIZE`, hard-capped at 25). Larger batches produce malformed JSON, such as missing commas or truncated arrays. Common JSON defects are repaired before parsing. If a call still fails after retries, that batch falls back to lexical grouping.

## Reddit

All Reddit requests use a fixed Chrome 122 browser User-Agent (`DEFAULT_HEADERS` in `ingestion/base.py`), a 3-second timeout and no retries. After the first 403 or 429 on a JSON listing, the remaining subreddits skip JSON and go straight to RSS. Subreddits are fetched 3 at a time, so a fully blocked Reddit costs about 10-15s rather than a minute.

## Velocity

For each entity: `rate = % of kept items mentioning it`. For each lookback window, the completed run closest to `now - w` (within ±50%) is found, and the rate is recomputed from that run's stored titles. `growth = (now - then) / max(then, one_item_floor)`. `velocity = 50 + 50 * weighted_mean(tanh(growth/2))`, with weights 0.5, 0.3 and 0.2 for 1h, 6h and 24h. 50 means flat. Momentum labels: NEW, SURGING (75+), RISING (60+), STEADY, COOLING (25+), FADING. Until history exists, the score is a cold-start estimate labelled `BASELINE`. Run in `--loop` mode to fill the 1h, 6h and 24h windows.

## Notes

- Titles are folded to ASCII (as specified), so non-Latin-script trends are discarded. For another region, change `AGENT_REACH_GEO`, `TRENDS24_REGION` and `WIKIPEDIA_PROJECT`.
- Trends24, GitHub Trending and TikTok are HTML scrapes. When their markup changes, the ingester reports `FAIL` in SOURCE HEALTH and the run continues.
- TikTok Creative Center often bot-gates unauthenticated requests. Expect intermittent `FAIL` from that source.
- Unreachable Ollama, a missing model or invalid JSON all fall back to deterministic clustering. The engine used is shown in the report header.

## Cloud runner (GitHub Actions)

`run_cloud_handoff.ps1` syncs this folder to `timeitself1-cpu/Agent-Reach`. It excludes secrets, blocks token-looking strings, and keeps remote-only files unless you pass `-Mirror`. It then dispatches `.github/workflows/cloud-runner.yml`, waits until the run is `in_progress`, and streams its logs.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cloud_handoff.ps1                 # full LLM run
.\run_cloud_handoff.ps1 -NoLLM -ExtraArgs "--top 10"                              # fast deterministic run
```

The workflow installs Ollama on a CPU-only `ubuntu-latest` runner and caches `llama3.1:8b` after the first pull. It caps the LLM batch at 80 items to keep runtime bounded. SQLite history is carried between runs in the Actions cache, so velocity works in the cloud as well. The report appears in the run summary and as a downloadable artifact. Optionally, set the repository variable `AGENT_REACH_CONTACT_EMAIL`.
