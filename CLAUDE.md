# Agent Reach: guide for Claude

Agent Reach is **Module 1** of a modular trend-intelligence system. It ingests trend signals from 10 sources by default (TikTok is opt-in), filters noise, enriches items with page content, groups them by embedding density, has a local LLM (`llama3.1:8b` via Ollama) label the groups, scores relevance and velocity, and writes an ASCII executive report plus SQLite history. Read `README.md` for the full architecture.

## Commands

```bash
pip install -r requirements-dev.txt     # runtime deps + pytest + pyflakes
python -m pytest                        # offline suite (~15 s): mock HTTP + fake Ollama, real HDBSCAN/trafilatura/SQLite
python -m pyflakes agent_reach tests    # must be clean
python -m agent_reach --no-llm --help   # CLI smoke test
```

Every change must leave `pytest` green and `pyflakes` clean. `.github/workflows/tests.yml` runs both (plus the CLI smoke test) on every push and pull request. Add or extend tests in `tests/` for any behaviour you change. `tests/fakes.py` holds the mock sources and the fake Ollama.

## Environment limits (cloud sessions)

- **No Ollama or live data here.** The cloud sandbox has no Ollama and cannot reach most source sites, so never try a live run. Verify with the offline tests.
- **Real-data runs happen in GitHub Actions.** The `.github/workflows/cloud-runner.yml` workflow is started manually (`workflow_dispatch`). It installs Ollama with `llama3.1:8b` and `nomic-embed-text` on a CPU runner and uploads `report.txt`, the JSON report and `pipeline.log` as an artifact. When a change needs real-data validation, say so in the PR description. The user will run the workflow and share the artifact.
- **`.ps1` scripts run on Windows PowerShell 5.1.** Keep `$ErrorActionPreference = "Continue"` with explicit `$LASTEXITCODE` checks, because 5.1 turns native stderr into terminating errors. Avoid PowerShell 7-only syntax (`??`, ternary `? :`, `&&`).

## Invariants: do not break these

1. **Item ledger balances.** `ingested == sum(discarded[reason]) + clustered`, in raw-item units (`CleanedTrendItem.raw_weight`). Every dropped item goes into exactly one named bucket (`models.DISCARD_STAGES`). `ClusterOutcome.assert_partition` and `PipelineAccounting` enforce this; tests assert it.
2. **The LLM never decides grouping.** Grouping comes from embeddings + HDBSCAN (`pipeline/density.py`), or the lexical fallback. The LLM only labels groups.
3. **Outliers are noise.** They are dropped, never forced into a mixed cluster.
4. **Entity isolation.** A cluster must be connected by distinctive shared tokens, a literally shared entity, or entities that co-occur in another item of the run (`LinkIndex`). An LLM entity list alone is never evidence.
5. **No filler.** `[INSUFFICIENT_DATA]`, placeholder summaries (`FILLER_RX`) and relevance <= 3 are dropped before the report.
6. **Ingesters never raise.** `BaseIngester.run()` returns `([], SourceStat(ok=False, ...))` on any failure. Reddit and TikTok use a 5 s timeout, 403/429 backoff retries and at least 2 s pacing.
7. **Output contract.** `PipelineReport` (`schema_version`) is consumed by future modules. Changing or removing fields requires bumping `schema_version`; adding optional fields does not.

## Conventions

- Python 3.10+ with `from __future__ import annotations`, Pydantic v2 and `httpx` (the only HTTP client). No LangGraph, aiohttp or other new frameworks without a clear need.
- Surgical, targeted changes over rewrites. Keep the module layout. Complete code only: no placeholders or TODOs.
- Report output is ASCII only. Titles and summaries go through `normalize_text`, `sanitize_summary` and `sanitize_headline`.
- Every behaviour-changing setting lives in `config.py` (`AGENT_REACH_*` env vars) with a safe default.
- Never commit `.env`, `*.db`, `.venv/`, `reports/` or `state/` (see `.gitignore`).

## Known open items

- Similarity thresholds (`density_member_min_cosine=0.55`, `hdbscan_selection="leaf"`) are defaults for `nomic-embed-text`. They have not been calibrated on real output yet. Use the `stage 3a density` log line from a cloud-runner artifact to tune them.
- Google News links are `news.google.com` redirect pages with no article text, so those items get no context.
- Reddit is usually blocked from data-centre IPs (CI). The robust fix is Reddit's OAuth API, with credentials as GitHub secrets.

## Roadmap

- **Module 2, Agent Depth:** reads `PipelineReport` JSON (the contract above), fetches 3-5 sources per top trend, checks cross-source agreement, and writes cited `TrendBrief` records. It must depend on the JSON contract only, never on Agent Reach imports.
- **Module 3:** delivery and watchlists (digest, velocity alerts).
- **Module 4:** actions.
