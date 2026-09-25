"""AI Papers of the Week: newest-week parsing, year fallback, retries, and never-raise failures."""
import asyncio
from datetime import datetime, timezone

import httpx

from agent_reach.config import Settings
from agent_reach.ingestion.papers import AIPapersOfTheWeekIngester, parse_latest_week
from agent_reach.models import CategoryEnum
from tests.fakes import PAPERS_MD, RealAsyncClient


def _run(handler, **settings):
    s = Settings(http_backoff_base_s=0.01, http_backoff_max_s=0.02, **settings)

    async def go():
        async with RealAsyncClient(transport=httpx.MockTransport(handler)) as cl:
            return await AIPapersOfTheWeekIngester(cl, s, asyncio.Semaphore(4)).run()

    return asyncio.run(go())


def test_parses_only_the_newest_week():
    label, week_end, rows = parse_latest_week(PAPERS_MD, 2026)
    assert label == "September 14 - September 20"
    assert week_end == datetime(2026, 9, 20, tzinfo=timezone.utc)
    assert [r["title"] for r in rows] == ["Agentic Reasoning Scaling", "GAUGE"]  # not last week's pick
    assert rows[0]["description"] == "A study of how agentic reasoning scales with model size and tool access."
    assert rows[0]["url"] == "https://arxiv.org/abs/2609.00001"  # arXiv id lifted from the paper link
    assert rows[1]["url"] == "https://example.org/gauge"


def test_ingester_builds_ranked_science_items():
    items, stat = _run(lambda req: httpx.Response(200, text=PAPERS_MD))
    assert stat.ok and stat.item_count == 2
    assert items[0].raw_score > items[1].raw_score
    assert all(it.category_hint is CategoryEnum.SCIENCE_AI for it in items)
    assert items[0].metadata["week"] == "September 14 - September 20"


def test_falls_back_to_previous_year_file():
    this_year = datetime.now(timezone.utc).year
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path.endswith(f"/{this_year}.md"):
            return httpx.Response(404)
        return httpx.Response(200, text=PAPERS_MD)

    items, stat = _run(handler)
    assert stat.ok and len(items) == 2
    assert seen[-1].endswith(f"/{this_year - 1}.md")


def test_retries_rate_limits_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] == 1 else httpx.Response(200, text=PAPERS_MD)

    items, stat = _run(handler)
    assert stat.ok and len(items) == 2 and calls["n"] == 2


def test_never_raises_when_nothing_parses():
    items, stat = _run(lambda req: httpx.Response(200, text="# moved elsewhere"))
    assert items == [] and not stat.ok and "no weekly paper table" in stat.error
