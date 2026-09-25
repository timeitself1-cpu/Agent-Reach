"""Source ingesters. ``build_ingesters`` returns the enabled set wired to a shared client."""

from __future__ import annotations

import asyncio

import httpx

from agent_reach.config import Settings
from agent_reach.ingestion.base import BaseIngester, IngestionError
from agent_reach.ingestion.papers import AIPapersOfTheWeekIngester
from agent_reach.ingestion.search import (
    ArxivIngester,
    GoogleNewsIngester,
    GoogleTrendsIngester,
    WikipediaPageviewsIngester,
)
from agent_reach.ingestion.social import RedditIngester, TikTokCreativeCenterIngester, XTrends24Ingester
from agent_reach.ingestion.tech import GitHubTrendingIngester, HackerNewsIngester, ProductHuntIngester

INGESTER_REGISTRY: dict[str, type[BaseIngester]] = {
    "x_trends24": XTrends24Ingester,
    "reddit": RedditIngester,
    "tiktok": TikTokCreativeCenterIngester,
    "google_trends": GoogleTrendsIngester,
    "google_news": GoogleNewsIngester,
    "wikipedia": WikipediaPageviewsIngester,
    "arxiv": ArxivIngester,
    "ai_papers": AIPapersOfTheWeekIngester,
    "hackernews": HackerNewsIngester,
    "github": GitHubTrendingIngester,
    "producthunt": ProductHuntIngester,
}


def build_ingesters(
    client: httpx.AsyncClient, settings: Settings, only: list[str] | None = None
) -> list[BaseIngester]:
    names = only or settings.enabled_sources
    unknown = [n for n in names if n not in INGESTER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}. Valid: {', '.join(INGESTER_REGISTRY)}")
    sem = asyncio.Semaphore(settings.http_max_concurrency)
    return [INGESTER_REGISTRY[n](client, settings, sem) for n in names]


__all__ = ["BaseIngester", "IngestionError", "INGESTER_REGISTRY", "build_ingesters"]
