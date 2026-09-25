"""Stage 2b: content enrichment.

Headline-only clustering forces the LLM to guess what a story is about. Before clustering,
every candidate item gets a short, factual ``context`` string built from its source page:
the page title, the meta/OpenGraph description and the first one or two substantive
paragraphs (trafilatura main-text extraction when installed, BeautifulSoup otherwise).

Per-source strategy (no wasted requests):
    arxiv            abstract already in the feed               -> feed
    ai_papers        curator's summary already in the feed      -> feed
    producthunt      tagline already in the feed (site is bot-walled) -> feed
    reddit           self-text, else the linked external article     -> feed / page
    google_trends    fetch the first linked news article, fall back to the news headlines
    wikipedia        REST summary API (clean lead extract)           -> wikipedia_api
    hackernews       linked article (Ask/Show HN without a URL: skipped)
    github           repository page (README lead)
    google_news      article page when the redirect resolves to the publisher
    x_trends24, tiktok   search/tag pages carry no article text -> no context

The HTTP client is the pipeline's shared ``httpx.AsyncClient``; every fetch is bounded by a
timeout, a byte cap and a concurrency limit, and failures simply leave ``context=None``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup

from agent_reach.config import Settings
from agent_reach.ingestion.base import DEFAULT_HEADERS
from agent_reach.models import CleanedTrendItem, SourceName
from agent_reach.pipeline.cleaner import normalize_text

log = logging.getLogger(__name__)

try:  # optional, better main-text extraction
    import trafilatura  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001 - any import problem => BeautifulSoup fallback
    trafilatura = None

NO_PAGE_SOURCES = frozenset({SourceName.X_TRENDS24, SourceName.TIKTOK})
FEED_CONTEXT_SOURCES = frozenset({SourceName.ARXIV, SourceName.AI_PAPERS, SourceName.PRODUCTHUNT})
BOILERPLATE_RX = re.compile(
    r"(cookie|subscribe|sign up|sign in|log in|newsletter|javascript|enable js|accept all|privacy policy|"
    r"all rights reserved|advertisement|skip to (main )?content|you have been blocked|access denied|"
    r"are you a robot|captcha|checking your browser)",
    re.IGNORECASE,
)
MIN_PARAGRAPH_CHARS = 60


@dataclass(frozen=True)
class _Fetched:
    url: str
    host: str
    content_type: str
    encoding: str
    body: bytes


def _is_public_http_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    host = p.hostname.lower()
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        return True


def _clip(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "..."


def extract_page_context(html: str, url: str | None, max_chars: int) -> tuple[str, str, list[str]]:
    """Return (page_title, meta_description, lead_paragraphs) from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = str(og_title["content"])
    elif soup.title and soup.title.string:
        title = soup.title.string
    meta = ""
    for attrs in ({"property": "og:description"}, {"name": "description"}, {"name": "twitter:description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            meta = str(tag["content"])
            break

    paragraphs: list[str] = []
    if trafilatura is not None:
        try:
            body = trafilatura.extract(
                html, url=url, include_comments=False, include_tables=False, favor_precision=True, deduplicate=True
            )
        except Exception:  # noqa: BLE001 - extraction must never break the stage
            body = None
        if body:
            paragraphs = [p.strip() for p in body.split("\n") if len(p.strip()) >= MIN_PARAGRAPH_CHARS]
    if not paragraphs:
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        scope = soup.find("article") or soup.find("main") or soup.body or soup
        for p in scope.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) >= MIN_PARAGRAPH_CHARS:
                paragraphs.append(text)
            if len(paragraphs) >= 4:
                break
    paragraphs = [p for p in paragraphs if not BOILERPLATE_RX.search(p[:160])]
    meta = "" if BOILERPLATE_RX.search(meta) else meta
    return normalize_text(title), normalize_text(meta), [normalize_text(p) for p in paragraphs[:2]]


def build_context(title: str, meta: str, paragraphs: list[str], item_title: str, max_chars: int) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    item_key = item_title.lower()[:60]
    for part in (title, meta, *paragraphs):
        part = part.strip()
        key = part.lower()[:80]
        if not part or key in seen or (part.lower()[:60] == item_key and len(part) < 120):
            continue
        # skip a meta description that the first paragraph already contains
        if any(key[:50] in s for s in seen):
            continue
        seen.add(key)
        parts.append(part)
    if not parts:
        return None
    ctx = _clip(" | ".join(parts), max_chars)
    return ctx if len(re.sub(r"[^A-Za-z]", "", ctx)) >= 40 else None


class ContentEnricher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats: Counter[str] = Counter()

    async def enrich(self, items: list[CleanedTrendItem], client: httpx.AsyncClient) -> dict[str, int]:
        """Populate ``item.context`` in place. Returns per-outcome counts."""
        self.stats = Counter()
        if not self.settings.enrich_enabled or not items:
            self.stats["disabled" if not self.settings.enrich_enabled else "empty"] += len(items)
            return dict(self.stats)
        started = time.perf_counter()
        sem = asyncio.Semaphore(self.settings.enrich_concurrency)

        async def one(it: CleanedTrendItem) -> None:
            try:
                outcome = await self._enrich_one(it, client, sem)
            except Exception as exc:  # noqa: BLE001 - one bad page never breaks the stage
                log.debug("enrich failed for %s: %s", it.url, exc)
                outcome = "error"
            self.stats[outcome] += 1

        await asyncio.gather(*(one(it) for it in items))
        with_ctx = sum(1 for it in items if it.context)
        self.stats["with_context"] = with_ctx
        log.info(
            "stage 2b enrich: %d/%d items have context in %.1f s (%s)",
            with_ctx, len(items), time.perf_counter() - started,
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items()) if k != "with_context"),
        )
        return dict(self.stats)

    async def _enrich_one(self, it: CleanedTrendItem, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> str:
        limit = self.settings.enrich_max_chars
        if it.source in NO_PAGE_SOURCES:
            return "no_page"
        if it.source in FEED_CONTEXT_SOURCES:
            if it.description:
                it.context, it.context_source = _clip(it.description, limit), "feed"
                return "feed"
            return "no_context"
        if it.source is SourceName.WIKIPEDIA:
            return await self._wikipedia(it, client, sem)
        if it.source is SourceName.REDDIT and it.description and len(it.description) >= MIN_PARAGRAPH_CHARS:
            it.context, it.context_source = _clip(it.description, limit), "feed"
            return "feed"

        url = it.url
        if it.source is SourceName.REDDIT:
            url = it.metadata.get("external_url") or None
        elif it.source is SourceName.GOOGLE_TRENDS:
            urls = it.metadata.get("news_urls") or []
            url = urls[0] if urls else None
        elif it.source is SourceName.HACKERNEWS and url and "news.ycombinator.com/item" in url:
            url = None  # Ask/Show HN without an external article

        if _is_public_http_url(url):
            ctx = await self._page(url, it.normalized_title, client, sem)
            if ctx:
                it.context, it.context_source = ctx, "page"
                return "page"
        # feed-level fallbacks
        if it.source is SourceName.GOOGLE_TRENDS and it.metadata.get("news_titles"):
            it.context = _clip(" | ".join(str(t) for t in it.metadata["news_titles"][:3]), limit)
            it.context_source = "feed"
            return "feed"
        if it.description and len(it.description) >= 30:
            it.context, it.context_source = _clip(it.description, limit), "feed"
            return "feed"
        return "fetch_failed" if url else "no_url"

    async def _get(self, url: str, client: httpx.AsyncClient, sem: asyncio.Semaphore, accept: str) -> _Fetched | None:
        """GET with timeout + byte cap; returns None on any HTTP/transport problem."""
        headers = dict(DEFAULT_HEADERS)
        headers["Accept"] = accept
        async with sem:
            try:
                async with client.stream(
                    "GET", url, headers=headers, timeout=self.settings.enrich_timeout_s, follow_redirects=True
                ) as resp:
                    if resp.status_code != 200:
                        return None
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= self.settings.enrich_max_bytes:
                            break
                    return _Fetched(
                        url=str(resp.url),
                        host=resp.url.host or "",
                        content_type=resp.headers.get("content-type", ""),
                        encoding=resp.charset_encoding or "utf-8",
                        body=b"".join(chunks),
                    )
            except (httpx.HTTPError, asyncio.TimeoutError):
                return None

    async def _page(self, url: str, item_title: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> str | None:
        got = await self._get(url, client, sem, "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5")
        if got is None:
            return None
        if "html" not in got.content_type and "xml" not in got.content_type:
            return None
        if got.host.endswith("news.google.com"):
            return None  # unresolved Google News redirect page: no article text
        try:
            html = got.body.decode(got.encoding, errors="replace")
        except LookupError:
            html = got.body.decode("utf-8", errors="replace")
        title, meta, paragraphs = await asyncio.to_thread(extract_page_context, html, got.url, self.settings.enrich_max_chars)
        return build_context(title, meta, paragraphs, item_title, self.settings.enrich_max_chars)

    async def _wikipedia(self, it: CleanedTrendItem, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> str:
        lang = self.settings.wikipedia_project.split(".")[0]
        page = (it.url or "").rsplit("/wiki/", 1)[-1] if it.url and "/wiki/" in it.url else quote(it.title.replace(" ", "_"))
        api = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{page}"
        got = await self._get(api, client, sem, "application/json")
        if got is None:
            return "fetch_failed"
        try:
            data = json.loads(got.body.decode("utf-8", errors="replace"))
        except ValueError:
            return "fetch_failed"
        if not isinstance(data, dict):
            return "fetch_failed"
        extract = str(data.get("extract") or "")
        desc = str(data.get("description") or "")
        ctx = build_context("", desc, [extract] if extract else [], it.normalized_title, self.settings.enrich_max_chars)
        if ctx:
            it.context, it.context_source = ctx, "wikipedia_api"
            return "wikipedia_api"
        return "no_context"
