"""Reach, the chat agent: answers questions about one trends report, grounded in it.

The model sees the report (ranked trends, scores, summaries, links, source health and the
headlines of earlier local runs) in its system prompt. When a question names a trend
("#2", "trend 3", or words from its headline), Reach first reads that trend's source pages
with the pipeline's own page extractor and adds the excerpts, so follow-up questions can go
beyond the one-paragraph summary. Replies stream from Ollama's /api/chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx

from agent_reach.chat.snapshot import Snapshot
from agent_reach.config import Settings
from agent_reach.pipeline.cleaner import display_sources, normalize_text, significant_tokens

log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
TREND_REF_RX = re.compile(r"(?:#\s*|\btrend\s+|\bnumber\s+|\bno\.\s*)(\d{1,2})\b", re.IGNORECASE)

SYSTEM_RULES = """You are Reach, the analyst behind Agent Reach, a trend-intelligence system. You chat with the user about the trends report below.

Rules:
- Answer ONLY from the REPORT and the ARTICLE EXCERPTS below. If they do not cover the question, say so plainly and suggest what the user could check instead. Never invent facts, numbers, dates, names or quotes.
- Refer to trends by their number, like #2. When you state something from a source, add its link.
- relevance is 1-10 (how significant the labelling model judged the story). velocity is 0-100 (how fast mentions grew against earlier runs; 50 means steady or no history). momentum is NEW (first seen this run), RISING, STEADY or COOLING.
- The report is a snapshot taken at the time shown. Say how old it is when the user asks about "now" or "today".
- EARLIER RUNS are older, separate runs. Only mention them for questions about change over time, and then say they come from an earlier run, never from the current report.
- Be concise: a short paragraph or a few bullets. No preamble."""


@dataclass
class Excerpt:
    rank: int
    url: str
    text: str


def render_report(snap: Snapshot) -> str:
    r = snap.report
    lines = [
        f"REPORT: {snap.origin_label}, started {r.started_at:%Y-%m-%d %H:%M} UTC ({snap.age_hours:.1f} hours ago), "
        f"engine {r.llm_mode}, {r.ingested_count} items ingested, {len(r.macro_clusters)} trends."
    ]
    if r.source_stats:
        ok = [f"{s.source} {s.item_count}" for s in r.source_stats if s.ok]
        bad = [f"{s.source} ({(s.error or 'failed')[:60]})" for s in r.source_stats if not s.ok]
        lines.append("Sources OK: " + ", ".join(ok) + (". Failed: " + "; ".join(bad) if bad else ""))
    lines.append("")
    lines.append("TRENDS (ranked):")
    if not r.macro_clusters:
        lines.append("(none: this run produced no trends)")
    for rank, c in enumerate(r.macro_clusters, start=1):
        lines.append(f"#{rank} [{c.category.value}] {normalize_text(c.headline)}")
        growth = [f"{w.window_hours:g}h {w.growth * 100:+.0f}%" for w in c.velocity_windows if w.growth is not None]
        lines.append(
            f"   relevance {c.relevance_score}/10 | velocity {c.velocity_score:.0f} {c.momentum}"
            + (f" ({', '.join(growth)})" if growth else "")
            + f" | {c.raw_item_count} items from {display_sources(c.sources) if c.sources else 'unknown sources'}"
        )
        lines.append(f"   summary: {normalize_text(c.summary)}")
        if c.primary_entities:
            lines.append(f"   entities: {', '.join(c.primary_entities)}")
        for u in c.source_urls[:4]:
            lines.append(f"   link: {u}")
    if snap.history:
        lines.append("")
        lines.append("EARLIER RUNS (local history, newest first):")
        for h in snap.history:
            lines.append(f"- {h['started_at']:%Y-%m-%d %H:%M} UTC: " + ("; ".join(h["headlines"]) or "no trends"))
    return "\n".join(lines)


def _default_fetcher(settings: Settings) -> Callable[[list[tuple[str, str]]], list[str | None]]:
    """Read pages with the enricher's extractor (timeouts, byte cap, public URLs only)."""
    from agent_reach.pipeline.enricher import ContentEnricher, _is_public_http_url

    deep = settings.model_copy(update={"enrich_max_chars": settings.chat_deep_read_chars})

    def fetch(pages: list[tuple[str, str]]) -> list[str | None]:
        async def go() -> list[str | None]:
            enricher = ContentEnricher(deep)
            sem = asyncio.Semaphore(4)
            async with httpx.AsyncClient(timeout=deep.enrich_timeout_s) as client:
                async def one(url: str, title: str) -> str | None:
                    if not _is_public_http_url(url):
                        return None
                    try:
                        return await enricher._page(url, title, client, sem)
                    except Exception:  # noqa: BLE001 - a bad page is just skipped
                        return None

                return list(await asyncio.gather(*(one(u, t) for u, t in pages)))

        return asyncio.run(go())

    return fetch


class ReachAgent:
    def __init__(
        self,
        settings: Settings,
        snapshot: Snapshot,
        *,
        http: httpx.Client | None = None,
        fetch_pages: Callable[[list[tuple[str, str]]], list[str | None]] | None = None,
    ) -> None:
        self.settings = settings
        self.snapshot = snapshot
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=settings.ollama_timeout_s, write=30.0, pool=5.0)
        )
        self.fetch_pages = fetch_pages or _default_fetcher(settings)
        self._page_cache: dict[str, str | None] = {}

    @property
    def model(self) -> str:
        return self.settings.chat_model or self.settings.ollama_model

    # ------------------------------------------------------------ grounding
    def referenced_trends(self, question: str, limit: int = 2) -> list[int]:
        """1-based ranks the question is about: explicit '#2' / 'trend 2', else headline/entity overlap."""
        n = len(self.snapshot.report.macro_clusters)
        ranks = [int(m) for m in TREND_REF_RX.findall(question) if 1 <= int(m) <= n]
        if not ranks:
            q_toks = significant_tokens(question)
            q_low = question.lower()
            scored = []
            for rank, c in enumerate(self.snapshot.report.macro_clusters, start=1):
                overlap = len(q_toks & significant_tokens(c.headline))
                named = any(len(e) >= 3 and e.lower() in q_low for e in c.primary_entities)
                if overlap >= 2 or named:
                    scored.append((overlap + (2 if named else 0), rank))
            ranks = [r for _, r in sorted(scored, reverse=True)]
        return list(dict.fromkeys(ranks))[:limit]

    def deep_read(self, ranks: list[int]) -> list[Excerpt]:
        clusters = self.snapshot.report.macro_clusters
        wanted: list[tuple[int, str, str]] = []
        for rank in ranks:
            c = clusters[rank - 1]
            urls = [u for u in c.source_urls if "news.google.com" not in u][: self.settings.chat_deep_read_urls]
            wanted.extend((rank, u, c.headline) for u in urls)
        todo = [(u, h) for _, u, h in wanted if u not in self._page_cache]
        if todo:
            for (u, _), text in zip(todo, self.fetch_pages(todo)):
                self._page_cache[u] = text
        return [Excerpt(rank, u, self._page_cache[u]) for rank, u, _ in wanted if self._page_cache.get(u)]

    def system_prompt(self, excerpts: list[Excerpt]) -> str:
        parts = [SYSTEM_RULES, "", render_report(self.snapshot)]
        if excerpts:
            parts += ["", "ARTICLE EXCERPTS (read just now from the trends' source links):"]
            parts += [f"[#{e.rank}] {e.url}\n{e.text}" for e in excerpts]
        return "\n".join(parts)

    # ------------------------------------------------------------ chat
    def _clean_history(self, messages: list[dict]) -> list[dict]:
        out = [
            {"role": m["role"], "content": str(m.get("content", ""))[:MAX_MESSAGE_CHARS]}
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
        ]
        return out[-(2 * self.settings.chat_history_turns + 1):]

    def respond(self, messages: list[dict]) -> Iterator[dict]:
        """Yield events: {"type": "status"|"token"|"done"|"error", "text": ...}."""
        history = self._clean_history(messages)
        if not history or history[-1]["role"] != "user":
            yield {"type": "error", "text": "Send a question first."}
            return
        question = history[-1]["content"]
        ranks = self.referenced_trends(question)
        excerpts: list[Excerpt] = []
        if ranks and self.settings.chat_deep_read_urls:
            yield {"type": "status", "text": "Reading sources for " + ", ".join(f"#{r}" for r in ranks) + "..."}
            excerpts = self.deep_read(ranks)
        yield {"type": "status", "text": f"Thinking ({self.model})..."}

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt(excerpts)}, *history],
            "stream": True,
            "options": {"temperature": self.settings.chat_temperature, "num_ctx": self.settings.chat_num_ctx},
            "keep_alive": self.settings.ollama_keep_alive,
        }
        host = self.settings.ollama_host
        try:
            with self.http.stream("POST", f"{host}/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    yield {"type": "error", "text": f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}"}
                    return
                if resp.status_code != 200:
                    resp.read()
                    yield {"type": "error", "text": f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"}
                    return
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        yield {"type": "error", "text": f"Ollama error: {data['error']}"}
                        return
                    chunk = (data.get("message") or {}).get("content") or ""
                    if chunk:
                        yield {"type": "token", "text": chunk}
                    if data.get("done"):
                        break
        except httpx.ConnectError:
            yield {"type": "error", "text": f"Ollama isn't reachable at {host}. Start it (ollama serve) and try again."}
            return
        except httpx.TimeoutException:
            yield {"type": "error", "text": f"Ollama took longer than {self.settings.ollama_timeout_s:.0f} s to answer."}
            return
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            yield {"type": "error", "text": f"Chat failed: {type(exc).__name__}: {exc}"}
            return
        yield {"type": "done", "text": ""}
