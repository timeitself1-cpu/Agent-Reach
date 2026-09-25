import asyncio
import time

import httpx
import pytest

from agent_reach import main as M
from agent_reach.config import Settings
from agent_reach.ingestion.social import RedditIngester, TikTokCreativeCenterIngester
from agent_reach.models import CleanedTrendItem, PipelineAccounting, SourceName
from agent_reach.pipeline import clusterer as C
from agent_reach.pipeline import density as D
from agent_reach.pipeline import enricher as EN
from tests.fakes import PAGES, FakeOllama, RealAsyncClient, fake_vec


def test_bs4_fallback_extracts_lead_and_skips_boilerplate(monkeypatch):
    monkeypatch.setattr(EN, "trafilatura", None)
    _, _, paras = EN.extract_page_context(PAGES["f-droid.org"], "https://f-droid.org/x", 600)
    assert paras and "F-Droid project released" in paras[0]
    assert not any("newsletter" in p.lower() for p in paras)


def test_trafilatura_extraction():
    _, _, paras = EN.extract_page_context(PAGES["f-droid.org"], "https://f-droid.org/x", 600)
    assert paras


def test_ssrf_guard():
    assert not EN._is_public_http_url("http://127.0.0.1/a")
    assert not EN._is_public_http_url("http://localhost/x")
    assert not EN._is_public_http_url("http://10.0.0.5/x")
    assert EN._is_public_http_url("https://example.com/a")


def test_threshold_fallback_without_sklearn(monkeypatch):
    monkeypatch.setattr(D, "HDBSCAN", None)
    items = [CleanedTrendItem(item_id=i, title=t, normalized_title=t, source=SourceName.HACKERNEWS, heuristic_score=0.5)
             for i, t in enumerate(["Packers beat Falcons", "Jordan Love Packers win", "Samsung fridge bricked", "F-Droid 2.0 released"], 1)]
    res = D.density_cluster(items, D._normalise([fake_vec(i.title) for i in items]), Settings())
    assert res.method.startswith("cosine-threshold")
    assert [sorted(g) for g in res.groups] == [[1, 2]]
    assert sorted(res.noise) == [3, 4]


def test_ledger_validator():
    with pytest.raises(Exception, match="stage 2 ledger broken"):
        PipelineAccounting(ingested=10, passed_filters=9, clustering_candidates=9, clustered=3, discarded={"pet_post": 2})
    ok = PipelineAccounting(ingested=10, passed_filters=9, clustering_candidates=7, clustered=3,
                            discarded={"pet_post": 1, "not_selected_budget": 2, "density_noise": 4})
    assert ok.balanced


def test_ollama_down_falls_back_and_balances(mock_http, settings, monkeypatch):
    class Down:
        async def list(self):
            raise httpx.ConnectError("down")

        async def embed(self, **k):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(C.SemanticClusterer, "_get_client", lambda self: Down())
    r = asyncio.run(M.run_once(settings))
    assert r.llm_mode.startswith("heuristic") and "lexical" in r.llm_mode
    assert r.accounting.balanced


def test_missing_embedding_model_uses_lexical_grouping(mock_http, settings, monkeypatch):
    class NoEmbed(FakeOllama):
        async def embed(self, **k):
            raise RuntimeError('model "nomic-embed-text" not found, try pulling it first')

    fake = NoEmbed()
    monkeypatch.setattr(C.SemanticClusterer, "_get_client", lambda self: fake)
    r = asyncio.run(M.run_once(settings))
    assert r.llm_mode.startswith("ollama") and "lexical" in r.llm_mode
    assert r.accounting.balanced


def test_all_sources_down_yields_empty_balanced_report(settings, fake_ollama, monkeypatch):
    def boom(req):
        raise httpx.ConnectError("boom")

    class Bad(RealAsyncClient):
        def __init__(self, *a, **k):
            k["transport"] = httpx.MockTransport(boom)
            super().__init__(*a, **k)

    monkeypatch.setattr(M.httpx, "AsyncClient", Bad)
    r = asyncio.run(M.run_once(settings))
    assert r.ingested_count == 0 and r.cluster_count == 0 and r.accounting.balanced


def test_reddit_retries_pacing_and_budget():
    stamps: list[float] = []
    state = {"n": 0}

    def rh(req):
        stamps.append(time.monotonic())
        if req.url.path.endswith(".json"):
            return httpx.Response(403, text="Blocked")
        state["n"] += 1
        if state["n"] % 2:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, content=b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Senate votes on budget deal</title><link href="https://reddit.com/1"/><category term="news"/></entry></feed>')

    s = Settings(reddit_subreddits=["a", "b", "c", "d"], reddit_request_spacing_s=0.3, http_backoff_base_s=0.05).model_copy(update={"reddit_budget_s": 2.0})

    async def go():
        async with RealAsyncClient(transport=httpx.MockTransport(rh)) as cl:
            t0 = time.monotonic()
            items, stat = await RedditIngester(cl, s, asyncio.Semaphore(8)).run()
            return items, stat, time.monotonic() - t0

    items, stat, dt = asyncio.run(go())
    assert stat.ok and items
    assert min(b - a for a, b in zip(stamps, stamps[1:])) >= 0.29
    assert dt < 4.5


def test_tiktok_paced_retries_then_clean_fail():
    stamps: list[float] = []

    def th(req):
        stamps.append(time.monotonic())
        return httpx.Response(429 if len(stamps) < 3 else 403)

    s = Settings(tiktok_request_spacing_s=0.3, http_backoff_base_s=0.05)

    async def go():
        async with RealAsyncClient(transport=httpx.MockTransport(th)) as cl:
            return await TikTokCreativeCenterIngester(cl, s, asyncio.Semaphore(8)).run()

    items, stat = asyncio.run(go())
    assert not stat.ok and items == [] and len(stamps) == 3
    assert min(b - a for a, b in zip(stamps, stamps[1:])) >= 0.29
