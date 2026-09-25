"""Developer & tech ingesters: Hacker News (Algolia), GitHub Trending, Product Hunt."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

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


# =====================================================================  Hacker News
class HackerNewsIngester(BaseIngester):
    """Front page + top stories of the last 24h from the Algolia HN API."""

    source = SourceName.HACKERNEWS
    API = "https://hn.algolia.com/api/v1/search"

    async def fetch(self) -> list[RawTrendItem]:
        since = int(time.time()) - 24 * 3600
        hits: dict[str, dict] = {}
        errors: list[str] = []
        queries = (
            {"tags": "front_page", "hitsPerPage": 50},
            {
                "tags": "story",
                "numericFilters": f"created_at_i>{since},points>{self.settings.hn_min_points}",
                "hitsPerPage": 50,
            },
        )
        for params in queries:
            try:
                data = await self.get_json(self.API, params=params)
            except IngestionError as exc:
                errors.append(str(exc))
                continue
            for h in (data or {}).get("hits", []):
                oid = str(h.get("objectID"))
                if oid and oid not in hits:
                    hits[oid] = h
        if not hits and errors:
            raise IngestionError("; ".join(errors)[:300])

        out: list[RawTrendItem] = []
        for oid, h in hits.items():
            title = h.get("title") or h.get("story_title") or ""
            created = h.get("created_at_i")
            item = self.make_item(
                title=title,
                raw_score=float(h.get("points") or 0),
                comment_count=int(h.get("num_comments") or 0),
                url=h.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                timestamp=datetime.fromtimestamp(created, tz=timezone.utc) if created else datetime.now(timezone.utc),
                category_hint=CategoryEnum.TECH,
                metadata={"hn_id": oid, "discussion_url": f"https://news.ycombinator.com/item?id={oid}"},
            )
            if item:
                out.append(item)
        out.sort(key=lambda it: it.raw_score or 0.0, reverse=True)
        return out


# =====================================================================  GitHub Trending
class GitHubTrendingIngester(BaseIngester):
    """Scrapes github.com/trending (daily). Title = 'owner/repo: description'."""

    source = SourceName.GITHUB

    async def fetch(self) -> list[RawTrendItem]:
        html = await self.get_text("https://github.com/trending", params={"since": "daily"})
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("article.Box-row")
        if not rows:
            raise IngestionError("GitHub trending markup not recognised (no article.Box-row)")
        out: list[RawTrendItem] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            a = row.select_one("h2 a") or row.select_one("h1 a")
            if not a or not a.get("href"):
                continue
            repo = a["href"].strip("/")
            desc_el = row.find("p")
            desc = desc_el.get_text(" ", strip=True) if desc_el else ""
            lang_el = row.select_one('[itemprop="programmingLanguage"]')
            stars_today = None
            total_stars = None
            for span in row.find_all("span"):
                txt = span.get_text(" ", strip=True)
                m = re.search(r"([\d,\.]+k?)\s+stars?\s+(today|this week|this month)", txt, re.I)
                if m:
                    stars_today = parse_count(m.group(1))
                    break
            star_link = row.select_one('a[href$="/stargazers"]')
            if star_link:
                total_stars = parse_count(star_link.get_text(strip=True))
            owner, _, name = repo.partition("/")
            readable = name.replace("-", " ").replace("_", " ")
            title = f"{repo}: {desc}" if desc else f"{repo} ({readable})"
            item = self.make_item(
                title=title[:300],
                raw_score=stars_today if stars_today is not None else 0.0,
                url=f"https://github.com/{repo}",
                timestamp=now,
                category_hint=CategoryEnum.TECH,
                description=desc or None,
                metadata={
                    "repo": repo,
                    "owner": owner,
                    "language": lang_el.get_text(strip=True) if lang_el else None,
                    "stars_today": stars_today,
                    "total_stars": total_stars,
                },
            )
            if item:
                out.append(item)
        return out


# =====================================================================  Product Hunt
class ProductHuntIngester(BaseIngester):
    """Product Hunt Atom feed (newest featured launches)."""

    source = SourceName.PRODUCTHUNT

    async def fetch(self) -> list[RawTrendItem]:
        root = await self.get_xml("https://www.producthunt.com/feed")
        entries = [el for el in root.iter() if xml_local(el.tag) in ("entry", "item")]
        if not entries:
            raise IngestionError("Product Hunt feed contained no entries")
        n = len(entries)
        out: list[RawTrendItem] = []
        for rank, entry in enumerate(entries, start=1):
            title = xml_child_text(entry, "title") or ""
            link = None
            for c in entry:
                if xml_local(c.tag) == "link":
                    link = c.get("href") or (c.text or "").strip() or link
            content = xml_child_text(entry, "content") or xml_child_text(entry, "description") or ""
            tagline = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)[:300] if content else ""
            tagline = re.sub(r"\s*(Discussion|Link)\s*\|?\s*", " ", tagline).strip()
            item = self.make_item(
                title=f"{title}: {tagline}" if tagline and len(title) < 60 else title,
                raw_score=float(n - rank + 1),
                url=link,
                timestamp=parse_datetime(xml_child_text(entry, "published") or xml_child_text(entry, "pubDate")),
                category_hint=CategoryEnum.TECH,
                description=tagline or None,
                metadata={"product": title, "rank": rank},
            )
            if item:
                out.append(item)
        return out
