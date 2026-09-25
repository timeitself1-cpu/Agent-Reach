"""Agent Reach orchestrator.

    python -m agent_reach.main                    # one run, print executive report
    python -m agent_reach.main --loop --interval 30
    python -m agent_reach.main --sources hackernews github --no-llm
    python -m agent_reach.main --json-out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import textwrap
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent_reach.config import Settings, get_settings
from agent_reach.ingestion import INGESTER_REGISTRY, build_ingesters
from agent_reach.models import PipelineReport, RawTrendItem, SourceStat
from agent_reach.pipeline.cleaner import TrendCleaner, normalize_text
from agent_reach.pipeline.clusterer import SemanticClusterer
from agent_reach.pipeline.scorer import TrendScorer
from agent_reach.storage.db import TrendDatabase

log = logging.getLogger("agent_reach")


# ====================================================================== pipeline
async def ingest_all(settings: Settings, only: list[str] | None) -> tuple[list[RawTrendItem], list[SourceStat]]:
    limits = httpx.Limits(max_connections=settings.http_max_concurrency * 2, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=settings.http_timeout_s) as client:
        ingesters = build_ingesters(client, settings, only)
        results = await asyncio.gather(*(ing.run() for ing in ingesters), return_exceptions=True)
    items: list[RawTrendItem] = []
    stats: list[SourceStat] = []
    for ing, res in zip(ingesters, results):
        if isinstance(res, BaseException):  # run() never raises, but be defensive
            stats.append(SourceStat(source=ing.source.value, ok=False, item_count=0, latency_ms=0, error=str(res)[:300]))
            continue
        got, stat = res
        items.extend(got)
        stats.append(stat)
    return items, stats


async def run_once(settings: Settings, *, only: list[str] | None = None, use_llm: bool = True) -> PipelineReport:
    t0 = time.perf_counter()
    now = time.time()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    db = TrendDatabase(settings.db_path)
    try:
        await asyncio.to_thread(db.begin_run, run_id, now)

        # 1. ingest ---------------------------------------------------------
        raw_items, source_stats = await ingest_all(settings, only)
        log.info("ingested %d raw items from %d sources", len(raw_items), len(source_stats))

        # 2. clean ----------------------------------------------------------
        cleaner = TrendCleaner(settings)
        cleaned, breakdown = cleaner.clean(raw_items)
        llm_batch = cleaner.select_for_llm(cleaned)

        # 3. cluster --------------------------------------------------------
        clusterer = SemanticClusterer(settings)
        if use_llm:
            clusters, discarded, mode = await clusterer.cluster(llm_batch)
        else:
            clusters, discarded = clusterer.heuristic_cluster(llm_batch)
            mode = "heuristic (--no-llm)"

        # 4. score ----------------------------------------------------------
        scorer = TrendScorer(settings, db)
        clusters, snapshots = await scorer.score(clusters, cleaned, run_id, now)

        report = PipelineReport(
            run_id=run_id,
            execution_time=round(time.perf_counter() - t0, 2),
            ingested_count=len(raw_items),
            filtered_count=len(cleaned),
            cluster_count=len(clusters),
            macro_clusters=clusters,
            started_at=datetime.fromtimestamp(now, tz=timezone.utc),
            llm_mode=mode,
            discarded_by_llm=discarded,
            source_stats=source_stats,
            filter_breakdown=breakdown,
        )

        # 5. persist --------------------------------------------------------
        await asyncio.to_thread(db.save_items, run_id, raw_items, cleaned, now)
        await asyncio.to_thread(db.save_clusters, run_id, clusters)
        await asyncio.to_thread(db.save_entity_snapshots, run_id, snapshots, now)
        report.execution_time = round(time.perf_counter() - t0, 2)
        await asyncio.to_thread(db.finish_run, report, time.time())
        await asyncio.to_thread(db.purge_older_than, settings.retention_days)
        return report
    finally:
        db.close()


# ====================================================================== report
def _bar(value: float, maximum: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, value / maximum)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render_report(report: PipelineReport, settings: Settings, top_n: int | None = None) -> str:
    W = settings.report_width
    top_n = top_n or settings.report_top_n
    rule, thin = "=" * W, "-" * W
    ind = "      "
    out: list[str] = [rule, " AGENT REACH :: TREND INTELLIGENCE EXECUTIVE REPORT".center(W).rstrip(), rule]
    out.append(
        f" Run {report.run_id} | {report.started_at:%Y-%m-%d %H:%M UTC} | {report.execution_time:.1f}s | "
        f"Engine: {report.llm_mode}"
    )
    out.append(
        f" Pipeline: {report.ingested_count} ingested -> {report.filtered_count} passed filters -> "
        f"{report.cluster_count} macro-clusters ({report.discarded_by_llm} items discarded at clustering)"
    )
    out.append(thin)
    out.append(" SOURCE HEALTH")
    for s in report.source_stats:
        status = "OK  " if s.ok else "FAIL"
        line = f"   {s.source:<14} {status} {s.item_count:>4} items {s.latency_ms:>6} ms"
        if s.error:
            line += "  " + textwrap.shorten(s.error, max(20, W - len(line) - 4), placeholder="...")
        out.append(line)
    noise = {k: v for k, v in sorted(report.filter_breakdown.items(), key=lambda kv: -kv[1]) if k != "kept"}
    if noise:
        out.append(thin)
        out.append(" NOISE FILTER")
        out.extend(
            textwrap.wrap(", ".join(f"{k}={v}" for k, v in noise.items()), W - 3, initial_indent="   ", subsequent_indent="   ")
        )
    out.append(rule)
    w_rel = settings.rank_relevance_weight
    out.append(f" TOP {min(top_n, len(report.macro_clusters))} TRENDS  (rank = {w_rel:.1f} x relevance + {1 - w_rel:.1f} x velocity)")
    out.append(rule)
    if not report.macro_clusters:
        out.append("   No high-signal macro-trends this run.")
    for rank, c in enumerate(report.macro_clusters[:top_n], start=1):
        out.append(" " + textwrap.shorten(f"#{rank:<3} [{c.category.value}] {normalize_text(c.headline)}", W - 1, placeholder="..."))
        out.append(
            f"{ind}REL {c.relevance_score:>2}/10 {_bar(c.relevance_score, 10)}  "
            f"VEL {c.velocity_score:5.1f} {_bar(c.velocity_score, 100)} {c.momentum:<8}  "
            f"SCORE {c.combined_score:5.1f}"
        )
        out.append(f"{ind}{c.raw_item_count} items | sources: {', '.join(c.sources)} | basis: {c.velocity_basis}")
        hist = [w for w in c.velocity_windows if w.growth is not None]
        if hist:
            out.append(ind + "trend: " + "  ".join(f"{w.window_hours:g}h {w.growth * 100:+.0f}%" for w in hist))
        out.extend(textwrap.wrap(normalize_text(c.summary), W - len(ind), initial_indent=ind, subsequent_indent=ind))
        if c.primary_entities:
            out.append(ind + textwrap.shorten("entities: " + ", ".join(c.primary_entities), W - len(ind), placeholder="..."))
        for u in c.source_urls[:2]:
            out.append(ind + "-> " + (u if len(u) <= W - len(ind) - 3 else u[: W - len(ind) - 6] + "..."))
        out.append(thin)
    mix = Counter(c.category.value for c in report.macro_clusters)
    if mix:
        out.append(" CATEGORY MIX: " + " | ".join(f"{k} {v}" for k, v in mix.most_common()))
    out.append(rule)
    return "\n".join(out)


# ====================================================================== CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent_reach", description="Agent Reach trend-intelligence pipeline")
    p.add_argument("--loop", action="store_true", help="run continuously")
    p.add_argument("--interval", type=float, default=30.0, help="minutes between runs in --loop mode (default 30)")
    p.add_argument("--sources", nargs="+", choices=sorted(INGESTER_REGISTRY), help="restrict to these sources")
    p.add_argument("--no-llm", action="store_true", help="skip Ollama, use deterministic clustering")
    p.add_argument("--db", type=Path, help="SQLite path (overrides AGENT_REACH_DB_PATH)")
    p.add_argument("--top", type=int, help="number of trends to print")
    p.add_argument("--json-out", type=Path, help="also write the full PipelineReport as JSON")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _cycle(settings: Settings, args: argparse.Namespace) -> None:
    report = await run_once(settings, only=args.sources, use_llm=not args.no_llm)
    print(render_report(report, settings, args.top), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        log.info("wrote JSON report to %s", args.json_out)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    settings = get_settings()
    if args.db:
        settings = settings.model_copy(update={"db_path": args.db})

    if not args.loop:
        await _cycle(settings, args)
        return 0

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: fall back to KeyboardInterrupt
            pass
    interval = max(1.0, args.interval) * 60
    while not stop.is_set():
        started = time.monotonic()
        try:
            await _cycle(settings, args)
        except Exception:  # noqa: BLE001 - keep the daemon alive
            log.exception("run failed; will retry next interval")
        wait = max(5.0, interval - (time.monotonic() - started))
        log.info("next run in %.0f s", wait)
        try:
            await asyncio.wait_for(stop.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass
    log.info("shutdown requested; exiting")
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
