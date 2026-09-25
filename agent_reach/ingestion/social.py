"""Social & viral ingesters: X (via Trends24), Reddit (JSON with RSS fallback), TikTok Creative Center."""

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup

from agent_reach.ingestion.base import (
    BaseIngester,
    IngestionError,
    parse_count,
    parse_datetime,
    xml_child_text,
    xml_local,
)
from agent_reach.models import CategoryEnum, RawTrendItem, SourceName

SUBREDDIT_CATEGORY: dict[str, CategoryEnum] = {
    "news": CategoryEnum.NEWS,
    "worldnews": CategoryEnum.NEWS,
    "politics": CategoryEnum.NEWS,
    "technology": CategoryEnum.TECH,
    "programming": CategoryEnum.TECH,
    "gadgets": CategoryEnum.TECH,
    "apple": CategoryEnum.TECH,
    "android": CategoryEnum.TECH,
    "science": CategoryEnum.SCIENCE_AI,
    "space": CategoryEnum.SCIENCE_AI,
    "machinelearning": CategoryEnum.SCIENCE_AI,
    "artificial": CategoryEnum.SCIENCE_AI,
    "localllama": CategoryEnum.SCIENCE_AI,
    "openai": CategoryEnum.SCIENCE_AI,
    "sports": CategoryEnum.SPORTS,
    "nfl": CategoryEnum.SPORTS,
    "nba": CategoryEnum.SPORTS,
    "soccer": CategoryEnum.SPORTS,
    "baseball": CategoryEnum.SPORTS,
    "hockey": CategoryEnum.SPORTS,
    "cfb": CategoryEnum.SPORTS,
    "formula1": CategoryEnum.SPORTS,
    "movies": CategoryEnum.ENTERTAINMENT,
    "television": CategoryEnum.ENTERTAINMENT,
    "music": CategoryEnum.ENTERTAINMENT,
    "entertainment": CategoryEnum.ENTERTAINMENT,
    "gaming": CategoryEnum.ENTERTAINMENT,
    "popculturechat": CategoryEnum.ENTERTAINMENT,
    "memes": CategoryEnum.INTERNET_CULTURE,
    "outoftheloop": CategoryEnum.INTERNET_CULTURE,
    "interestingasfuck": CategoryEnum.INTERNET_CULTURE,
    "damnthatsinteresting": CategoryEnum.INTERNET_CULTURE,
}


# =====================================================================  X / Trends24
class XTrends24Ingester(BaseIngester):
    """Scrapes the most recent hourly trend card from trends24.in."""

    source = SourceName.X_TRENDS24

    async def fetch(self) -> list[RawTrendItem]:
        region = self.settings.trends24_region.strip("/")
        url = f"https://trends24.in/{region}/"
        html = await self.get_text(url)
        soup = BeautifulSoup(html, "html.parser")

        # Layout A (current): <div class="list-container"> ... <ol class="trend-card__list"><li>...
        lists = soup.select("ol.trend-card__list")
        entries: list[tuple[str, float | None]] = []
        if lists:
            for li in lists[0].find_all("li"):
                a = li.select_one("a.trend-link") or li.find("a")
                if not a:
                    continue
                name = a.get_text(" ", strip=True)
                count_el = li.select_one(".tweet-count")
                count = None
                if count_el is not None:
                    count = parse_count(count_el.get("data-count") or count_el.get_text(strip=True))
                entries.append((name, count))
        # Layout B (fallback): any search links to X/Twitter
        if not entries:
            seen: set[str] = set()
            for a in soup.select('a[href*="twitter.com/search"], a[href*="x.com/search"]'):
                name = a.get_text(" ", strip=True)
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    entries.append((name, None))
        if not entries:
            raise IngestionError("trends24 markup not recognised (no trend lists found)")

        n = len(entries)
        now = datetime.now(timezone.utc)
        items: list[RawTrendItem] = []
        for rank, (name, count) in enumerate(entries, start=1):
            # rank-based score when volume is missing: #1 -> n, last -> 1
            score = count if count is not None else float(n - rank + 1)
            item = self.make_item(
                title=name,
                raw_score=score,
                url=f"https://x.com/search?q={quote(name)}",
                timestamp=now,
                metadata={"rank": rank, "tweet_volume": count},
            )
            if item:
                items.append(item)
        return items


# =====================================================================  Reddit
class RedditIngester(BaseIngester):
    """Top posts of the day per subreddit.

    Uses the public JSON listing (score + comment counts, required by the noise filter) with a
    fixed modern browser User-Agent; Reddit answers generic/bot UAs with ``403 Blocked``.
    Each endpoint gets a short timeout and no retries so a blocked JSON call falls through to
    RSS in ~3 s instead of burning the full backoff budget. After the first 403/429 on JSON the
    remaining subreddits skip JSON entirely (circuit breaker). RSS items carry no engagement
    metrics; they are tagged ``metrics_verified=False`` with their rank in the day's top listing.
    """

    source = SourceName.REDDIT
    use_bot_user_agent = False
    rotate_user_agent = False  # always send DEFAULT_HEADERS verbatim

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._json_blocked = False

    async def fetch(self) -> list[RawTrendItem]:
        subs = self.settings.reddit_subreddits
        per_sub = max(5, self.settings.max_items_per_source // max(1, len(subs)))
        gate = asyncio.Semaphore(self.settings.reddit_concurrency)

        async def one(sub: str) -> tuple[list[RawTrendItem], str | None]:
            async with gate:
                await asyncio.sleep(random.uniform(0.1, 0.6))  # de-synchronise bursts
                if not self._json_blocked:
                    try:
                        return await self._fetch_json(sub, per_sub), None
                    except IngestionError as exc:
                        if any(code in str(exc) for code in ("403", "429", "Blocked")):
                            self._json_blocked = True
                        self.log.info("r/%s JSON failed (%s); falling back to RSS", sub, str(exc)[:120])
                try:
                    return await self._fetch_rss(sub, per_sub), None
                except IngestionError as exc:
                    return [], f"r/{sub}: {exc}"

        results = await asyncio.gather(*(one(sub) for sub in subs))
        items: list[RawTrendItem] = []
        errors: list[str] = []
        for got, err in results:
            items.extend(got)
            if err:
                errors.append(err)
        if not items and errors:
            raise IngestionError("; ".join(errors)[:300])
        if errors:
            self.log.info("%d/%d subreddits failed: %s", len(errors), len(subs), "; ".join(errors)[:200])
        # verified (JSON) items first by score, then RSS items by rank
        items.sort(
            key=lambda it: (
                bool(it.metadata.get("metrics_verified")),
                it.raw_score or 0.0,
                -int(it.metadata.get("rank") or 999),
            ),
            reverse=True,
        )
        return items

    async def _fetch_json(self, sub: str, limit: int) -> list[RawTrendItem]:
        data = await self.get_json(
            f"https://www.reddit.com/r/{sub}/top.json",
            params={"t": "day", "limit": min(100, limit * 2), "raw_json": 1},
            timeout=self.settings.reddit_timeout_s,
            max_retries=0,
        )
        children = (data or {}).get("data", {}).get("children", [])
        hint = SUBREDDIT_CATEGORY.get(sub.lower())
        out: list[RawTrendItem] = []
        for child in children:
            d: dict[str, Any] = child.get("data", {})
            if d.get("stickied") or d.get("over_18") or d.get("promoted"):
                continue
            subreddit = str(d.get("subreddit", sub))
            item = self.make_item(
                title=d.get("title", ""),
                raw_score=float(d.get("score") or 0),
                comment_count=int(d.get("num_comments") or 0),
                url=f"https://www.reddit.com{d.get('permalink', '')}",
                timestamp=datetime.fromtimestamp(float(d.get("created_utc") or 0), tz=timezone.utc)
                if d.get("created_utc")
                else datetime.now(timezone.utc),
                category_hint=SUBREDDIT_CATEGORY.get(subreddit.lower(), hint),
                description=(d.get("selftext") or "")[:500] or None,
                metadata={
                    "subreddit": subreddit,
                    "metrics_verified": True,
                    "is_self": bool(d.get("is_self")),
                    "post_hint": d.get("post_hint"),
                    "link_flair": d.get("link_flair_text"),
                    "external_url": d.get("url_overridden_by_dest"),
                },
            )
            if item:
                out.append(item)
            if len(out) >= limit:
                break
        return out

    async def _fetch_rss(self, sub: str, limit: int) -> list[RawTrendItem]:
        root = await self.get_xml(
            f"https://www.reddit.com/r/{sub}/top/.rss",
            params={"t": "day", "limit": limit},
            timeout=self.settings.reddit_timeout_s,
            max_retries=0,
        )
        hint = SUBREDDIT_CATEGORY.get(sub.lower())
        out: list[RawTrendItem] = []
        entries = [el for el in root.iter() if xml_local(el.tag) == "entry"]
        for rank, entry in enumerate(entries[:limit], start=1):
            link = next((c.get("href") for c in entry if xml_local(c.tag) == "link"), None)
            category = next((c.get("term") for c in entry if xml_local(c.tag) == "category"), sub)
            item = self.make_item(
                title=xml_child_text(entry, "title") or "",
                raw_score=None,
                url=link,
                timestamp=parse_datetime(xml_child_text(entry, "updated") or xml_child_text(entry, "published")),
                category_hint=SUBREDDIT_CATEGORY.get(str(category).lower(), hint),
                metadata={"subreddit": category, "metrics_verified": False, "rank": rank},
            )
            if item:
                out.append(item)
        return out


# =====================================================================  TikTok
class TikTokCreativeCenterIngester(BaseIngester):
    """Trending hashtags from TikTok Creative Center.

    The public page embeds its initial data in ``__NEXT_DATA__``; we walk that JSON
    for hashtag records rather than depending on a signed private API.
    """

    source = SourceName.TIKTOK
    PAGE = "https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en"

    async def fetch(self) -> list[RawTrendItem]:
        html = await self.get_text(self.PAGE, params={"countryCode": self.settings.geo, "period": 7})
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        records: list[dict[str, Any]] = []
        if script and script.string:
            try:
                records = list(self._walk(json.loads(script.string)))
            except json.JSONDecodeError as exc:
                raise IngestionError(f"TikTok __NEXT_DATA__ invalid JSON: {exc}") from exc
        if not records:
            # markup fallback: hashtag cards render '# name' in spans
            for span in soup.find_all(string=re.compile(r"^#\s?\w{2,}")):
                records.append({"hashtagName": str(span).strip().lstrip("# ").strip()})
        if not records:
            raise IngestionError("TikTok Creative Center returned no hashtag data (likely bot-gated)")

        seen: set[str] = set()
        items: list[RawTrendItem] = []
        n = len(records)
        for rank, rec in enumerate(records, start=1):
            name = str(rec.get("hashtagName") or rec.get("hashtag_name") or "").strip().lstrip("#")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            views = rec.get("videoViews") or rec.get("video_views") or rec.get("publishCnt") or rec.get("publish_cnt")
            try:
                score = float(views) if views is not None else float(n - rank + 1)
            except (TypeError, ValueError):
                score = float(n - rank + 1)
            industry = rec.get("industryInfo") or rec.get("industry_info") or {}
            item = self.make_item(
                title=f"#{name}",
                raw_score=score,
                url=f"https://www.tiktok.com/tag/{quote(name)}",
                category_hint=CategoryEnum.INTERNET_CULTURE,
                metadata={
                    "rank": rec.get("rank", rank),
                    "industry": industry.get("value") if isinstance(industry, dict) else None,
                    "is_promoted": bool(rec.get("isPromoted") or rec.get("is_promoted")),
                },
            )
            if item and not item.metadata.get("is_promoted"):
                items.append(item)
        return items

    @classmethod
    def _walk(cls, node: Any):
        if isinstance(node, dict):
            if "hashtagName" in node or "hashtag_name" in node:
                yield node
                return
            for v in node.values():
                yield from cls._walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from cls._walk(v)
