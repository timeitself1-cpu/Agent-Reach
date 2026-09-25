"""Search & news ingesters: Google Trends RSS, Google News RSS, Wikipedia Pageviews, ArXiv."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from agent_reach.ingestion.base import (
    BaseIngester,
    IngestionError,
    parse_count,
    parse_datetime,
    xml_child_text,
    xml_local,
)
from agent_reach.models import CategoryEnum, RawTrendItem, SourceName


# =====================================================================  Google Trends
class GoogleTrendsIngester(BaseIngester):
    """Daily trending searches RSS (includes approx traffic + linked news headlines)."""

    source = SourceName.GOOGLE_TRENDS
    URLS = (
        "https://trends.google.com/trending/rss",
        "https://trends.google.com/trends/trendingsearches/daily/rss",  # legacy endpoint
    )

    async def fetch(self) -> list[RawTrendItem]:
        last_exc: Exception | None = None
        for url in self.URLS:
            try:
                root = await self.get_xml(url, params={"geo": self.settings.geo})
                items = self._parse(root)
                if items:
                    return items
            except IngestionError as exc:
                last_exc = exc
        raise IngestionError(f"Google Trends RSS unavailable: {last_exc}")

    def _parse(self, root) -> list[RawTrendItem]:
        out: list[RawTrendItem] = []
        for el in root.iter():
            if xml_local(el.tag) != "item":
                continue
            title = xml_child_text(el, "title") or ""
            traffic = parse_count(xml_child_text(el, "approx_traffic"))
            news_titles: list[str] = []
            news_urls: list[str] = []
            for child in el:
                if xml_local(child.tag) == "news_item":
                    t = xml_child_text(child, "news_item_title")
                    u = xml_child_text(child, "news_item_url")
                    if t:
                        news_titles.append(t)
                    if u:
                        news_urls.append(u)
            item = self.make_item(
                title=title,
                raw_score=traffic,
                url=news_urls[0] if news_urls else f"https://www.google.com/search?q={quote(title)}",
                timestamp=parse_datetime(xml_child_text(el, "pubDate")),
                description=" | ".join(news_titles[:3]) or None,
                metadata={"news_titles": news_titles[:3], "news_urls": news_urls[:3], "approx_traffic": traffic},
            )
            if item:
                out.append(item)
        return out


# =====================================================================  Google News
class GoogleNewsIngester(BaseIngester):
    """Top stories RSS. Google appends ' - Publisher' to titles; we split that off."""

    source = SourceName.GOOGLE_NEWS

    async def fetch(self) -> list[RawTrendItem]:
        geo = self.settings.geo.upper()
        root = await self.get_xml(
            "https://news.google.com/rss",
            params={"hl": f"en-{geo}", "gl": geo, "ceid": f"{geo}:en"},
        )
        out: list[RawTrendItem] = []
        entries = [el for el in root.iter() if xml_local(el.tag) == "item"]
        n = len(entries)
        for rank, el in enumerate(entries, start=1):
            raw_title = xml_child_text(el, "title") or ""
            publisher = xml_child_text(el, "source")
            title = raw_title
            if publisher and raw_title.endswith(f" - {publisher}"):
                title = raw_title[: -len(publisher) - 3]
            else:
                title = re.sub(r"\s+-\s+[^-]{2,60}$", "", raw_title)
            item = self.make_item(
                title=title,
                raw_score=float(n - rank + 1),
                url=xml_child_text(el, "link"),
                timestamp=parse_datetime(xml_child_text(el, "pubDate")),
                category_hint=CategoryEnum.NEWS,
                metadata={"publisher": publisher, "rank": rank},
            )
            if item:
                out.append(item)
        return out


# =====================================================================  Wikipedia
WIKI_EXCLUDE = re.compile(
    r"^(Main_Page|Special:|Wikipedia:|File:|Portal:|Help:|Talk:|Category:|Template:|User:|Draft:|Module:|"
    r"Deaths_in_|List_of_|Search$|Undefined$|-$|\d{4}_in_|\d{4}$)",
    re.IGNORECASE,
)


class WikipediaPageviewsIngester(BaseIngester):
    """Top viewed articles for the most recent complete UTC day."""

    source = SourceName.WIKIPEDIA
    use_bot_user_agent = True

    async def fetch(self) -> list[RawTrendItem]:
        project = self.settings.wikipedia_project
        today = datetime.now(timezone.utc).date()
        last_exc: Exception | None = None
        for days_back in (1, 2):
            day = today - timedelta(days=days_back)
            url = (
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
                f"{project}/all-access/{day.year:04d}/{day.month:02d}/{day.day:02d}"
            )
            try:
                data = await self.get_json(url)
            except IngestionError as exc:
                last_exc = exc
                continue
            articles = ((data or {}).get("items") or [{}])[0].get("articles", [])
            return self._parse(articles, day)
        raise IngestionError(f"Wikipedia pageviews unavailable: {last_exc}")

    def _parse(self, articles: list[dict], day) -> list[RawTrendItem]:
        out: list[RawTrendItem] = []
        ts = datetime(day.year, day.month, day.day, 23, 59, tzinfo=timezone.utc)
        lang = self.settings.wikipedia_project.split(".")[0]
        for a in articles:
            name = str(a.get("article", ""))
            if not name or WIKI_EXCLUDE.search(name):
                continue
            item = self.make_item(
                title=name.replace("_", " "),
                raw_score=float(a.get("views") or 0),
                url=f"https://{lang}.wikipedia.org/wiki/{quote(name)}",
                timestamp=ts,
                metadata={"rank": a.get("rank"), "views": a.get("views"), "date": day.isoformat()},
            )
            if item:
                out.append(item)
            if len(out) >= self.settings.max_items_per_source:
                break
        return out


# =====================================================================  ArXiv
class ArxivIngester(BaseIngester):
    """Newest submissions in configured AI/ML categories (Atom API)."""

    source = SourceName.ARXIV
    use_bot_user_agent = True

    async def fetch(self) -> list[RawTrendItem]:
        query = " OR ".join(f"cat:{c}" for c in self.settings.arxiv_categories)
        root = await self.get_xml(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": self.settings.max_items_per_source,
            },
        )
        entries = [el for el in root if xml_local(el.tag) == "entry"]
        n = len(entries)
        out: list[RawTrendItem] = []
        for rank, entry in enumerate(entries, start=1):
            title = re.sub(r"\s+", " ", xml_child_text(entry, "title") or "")
            summary = re.sub(r"\s+", " ", xml_child_text(entry, "summary") or "")
            link = xml_child_text(entry, "id")
            for c in entry:
                if xml_local(c.tag) == "link" and c.get("rel") == "alternate":
                    link = c.get("href") or link
            categories = [c.get("term") for c in entry if xml_local(c.tag) == "category" and c.get("term")]
            authors = [xml_child_text(a, "name") for a in entry if xml_local(a.tag) == "author"]
            item = self.make_item(
                title=title,
                raw_score=float(n - rank + 1),
                url=link,
                timestamp=parse_datetime(xml_child_text(entry, "published")),
                category_hint=CategoryEnum.SCIENCE_AI,
                description=summary[:500] or None,
                metadata={"categories": categories, "authors": [a for a in authors if a][:5]},
            )
            if item:
                out.append(item)
        return out
