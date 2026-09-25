"""Stage 3: density clustering, LLM labelling and entity isolation (llama3.1:8b via Ollama).

Flow
----
3a  Density grouping: embed title + enriched context (nomic-embed-text), cluster with HDBSCAN,
    gate every member by cosine-to-centroid. Outliers are NOISE and are dropped, never forced
    into a mixed bucket. (No embeddings available -> lexical union-find groups, same rule.)
3b  LLM labelling: the model only NAMES groups (headline, category, entities, 2-sentence
    summary, relevance); it never decides grouping. Groups whose signals cannot explain what
    happened and why are flagged [INSUFFICIENT_DATA].
3c  Entity isolation: each group must be a connected graph of items that share distinctive
    tokens, name the same entity, or name entities that co-occur elsewhere in the run.
    Mixed groups are split; split parts are re-labelled; unsupported strays are dropped.
3d  Deterministic merge of groups that resolve to the same entities.
3e  Finalise: drop [INSUFFICIENT_DATA] / filler summaries, relevance <= 3 and weak singletons;
    category guardrails; Title Case headlines; exactly two clean sentences.

Every dropped item is recorded under a discard reason so stage accounting always balances.
If Ollama is unreachable the same flow runs with heuristic labels, so the pipeline never hard-fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from agent_reach.config import Settings
from agent_reach.models import (
    CategoryEnum,
    CleanedTrendItem,
    MacroCluster,
    SourceName,
    _coerce_ids,
)
from agent_reach.pipeline.density import EmbeddingUnavailable, density_cluster, embed_items
from agent_reach.pipeline.cleaner import (
    STOPWORDS,
    category_votes,
    dedupe_key,
    display_sources,
    is_generic_headline,
    normalize_text,
    sanitize_headline,
    sanitize_summary,
    significant_tokens,
)

log = logging.getLogger(__name__)

#: llama3.1:8b starts dropping commas / truncating JSON above ~25 items per call.
MAX_BATCH_SIZE = 25
MIN_BATCH_SIZE = 5
HEADLINE_MAX_WORDS = 10

CATEGORY_ALIASES: dict[str, CategoryEnum] = {
    "sports": CategoryEnum.SPORTS,
    "sport": CategoryEnum.SPORTS,
    "athletics": CategoryEnum.SPORTS,
    "entertainment": CategoryEnum.ENTERTAINMENT,
    "movies": CategoryEnum.ENTERTAINMENT,
    "film": CategoryEnum.ENTERTAINMENT,
    "tv": CategoryEnum.ENTERTAINMENT,
    "television": CategoryEnum.ENTERTAINMENT,
    "music": CategoryEnum.ENTERTAINMENT,
    "gaming": CategoryEnum.ENTERTAINMENT,
    "games": CategoryEnum.ENTERTAINMENT,
    "celebrity": CategoryEnum.ENTERTAINMENT,
    "tech": CategoryEnum.TECH,
    "technology": CategoryEnum.TECH,
    "software": CategoryEnum.TECH,
    "hardware": CategoryEnum.TECH,
    "developer": CategoryEnum.TECH,
    "programming": CategoryEnum.TECH,
    "cybersecurity": CategoryEnum.TECH,
    "security": CategoryEnum.TECH,
    "news": CategoryEnum.NEWS,
    "politics": CategoryEnum.NEWS,
    "world": CategoryEnum.NEWS,
    "world news": CategoryEnum.NEWS,
    "business": CategoryEnum.NEWS,
    "finance": CategoryEnum.NEWS,
    "economy": CategoryEnum.NEWS,
    "weather": CategoryEnum.NEWS,
    "crime": CategoryEnum.NEWS,
    "health": CategoryEnum.NEWS,
    "internet culture": CategoryEnum.INTERNET_CULTURE,
    "internet": CategoryEnum.INTERNET_CULTURE,
    "culture": CategoryEnum.INTERNET_CULTURE,
    "memes": CategoryEnum.INTERNET_CULTURE,
    "meme": CategoryEnum.INTERNET_CULTURE,
    "social media": CategoryEnum.INTERNET_CULTURE,
    "viral": CategoryEnum.INTERNET_CULTURE,
    "lifestyle": CategoryEnum.INTERNET_CULTURE,
    "science & ai": CategoryEnum.SCIENCE_AI,
    "science and ai": CategoryEnum.SCIENCE_AI,
    "science/ai": CategoryEnum.SCIENCE_AI,
    "science": CategoryEnum.SCIENCE_AI,
    "ai": CategoryEnum.SCIENCE_AI,
    "artificial intelligence": CategoryEnum.SCIENCE_AI,
    "research": CategoryEnum.SCIENCE_AI,
    "space": CategoryEnum.SCIENCE_AI,
    "machine learning": CategoryEnum.SCIENCE_AI,
}

_WRITING_RULES = """HEADLINE rules: an authoritative, specific Title Case title of at most 10 words that names the concrete entity or event (e.g. "Packers Edge Falcons on Thursday Night Football", "OpenAI Releases GPT-6 With Native Agents"). Never write generic umbrella titles such as "Entertainment: Music and Film", "Tech: Tools and Innovations" or "Global Events and Diplomacy". Do not prefix the headline with the category name.
SUMMARY rules: exactly TWO complete, grammatically correct sentences in active voice. Sentence 1 states what happened and who did it. Sentence 2 explains why it is drawing attention, using the provided context. Use ONLY facts present in the titles and context; never invent numbers, dates, scores or quotes. Never include URLs, @handles, hashtags, emoji, markdown or JSON fragments. Never write filler such as "this has significant implications", "worth monitoring", "no specific information is available" or "details are scarce".
CATEGORY rules: exactly one of Sports, Entertainment, Tech, News, Internet Culture, Science & AI. Tech = software, hardware, developer tools, startups, tech companies, cybersecurity. Science & AI = AI models and research, scientific discoveries, space, research papers. Never label sports, celebrities, politics, pets or memes as Tech.
PRIMARY_ENTITIES: 1-5 proper nouns (people, teams, organizations, products) that appear in the group's own signals.
RELEVANCE_SCORE: integer 1-10 for significance and breadth of interest (10 = major global story, 1 = trivial)."""

INSUFFICIENT_FLAG = "[INSUFFICIENT_DATA]"

CLUSTER_SYSTEM_PROMPT = f"""You are the labelling and summarisation engine of Agent Reach, a real-time trend-intelligence system.
Signals have ALREADY been grouped by a density-based clustering step. Do NOT move signals between groups and do NOT merge groups.
Each signal line reads: source | category hint | title | context. The context is text scraped from the linked page (page title, meta description, lead paragraph) when it was available.

For each numbered group write headline, category, primary_entities, summary and relevance_score describing ONLY that group's signals.

INSUFFICIENT DATA RULE (strict): if the titles and context of a group do not let you state concretely what happened AND why it is drawing attention, set that group's summary to exactly {INSUFFICIENT_FLAG} and its relevance_score to 1. Do not guess and do not pad.

If unassigned signals are listed, give for each one the group_id of the group it clearly belongs to because it names the same specific person, team, organization, product or event as that group; otherwise use group_id 0.

{_WRITING_RULES}

Respond with JSON only."""

#: backwards-compatible alias (the labelling prompt is the only prompt now)
RELABEL_PROMPT = CLUSTER_SYSTEM_PROMPT

INSUFFICIENT_RX = re.compile(r"\[?\s*insufficient[\s_-]*data\s*\]?", re.IGNORECASE)
FILLER_RX = re.compile(
    r"\b(?:no (?:specific|further|additional|detailed) (?:information|details)|details (?:are|remain) (?:scarce|unclear|limited)|"
    r"(?:not|in)sufficient (?:information|context|data|details)|not enough (?:information|context|details)|"
    r"limited information|information is limited|unclear why|it is unclear (?:what|why)|cannot (?:be )?determine[d]?|"
    r"no (?:clear|further) context|without (?:more|further|additional) (?:information|context))\b",
    re.IGNORECASE,
)


# ======================================================================= schemas
def _relabel_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "integer"},
                        "headline": {"type": "string"},
                        "category": {"type": "string", "enum": CategoryEnum.values()},
                        "primary_entities": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "relevance_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["group_id", "headline", "category", "primary_entities", "summary", "relevance_score"],
                },
            },
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}, "group_id": {"type": "integer"}},
                    "required": ["item_id", "group_id"],
                },
            },
        },
        "required": ["groups", "assignments"],
    }


class _RelabelGroup(BaseModel):
    group_id: int
    headline: str = ""
    category: str = ""
    primary_entities: list[str] = Field(default_factory=list)
    summary: str = ""
    relevance_score: int = 5

    @field_validator("group_id", mode="before")
    @classmethod
    def _gid(cls, v: Any) -> int:
        ids = _coerce_ids(v)
        return ids[0] if ids else -1

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _rel(cls, v: Any) -> int:
        try:
            return max(1, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            return 5


class _RelabelAssignment(BaseModel):
    item_id: int
    group_id: int

    @field_validator("item_id", "group_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        ids = _coerce_ids(v)
        return ids[0] if ids else -1


class _RelabelResponse(BaseModel):
    groups: list[_RelabelGroup] = Field(default_factory=list)
    assignments: list[_RelabelAssignment] = Field(default_factory=list)


class ClusteringError(Exception):
    pass


# ======================================================================= helpers
def coerce_category(raw: str | None, fallback: CategoryEnum = CategoryEnum.NEWS) -> CategoryEnum:
    if not raw:
        return fallback
    key = normalize_text(str(raw)).strip().lower()
    for c in CategoryEnum:
        if key == c.value.lower():
            return c
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in key:
            return cat
    return fallback


def _repair_json(text: str) -> str:
    """Fix the most common 8B-model JSON defects: trailing commas and missing commas between items."""
    t = re.sub(r",\s*([}\]])", r"\1", text)
    t = re.sub(r"([}\]\"0-9])\s*\n\s*([{\"\[])", r"\1,\n\2", t)
    t = re.sub(r"}\s*{", "},{", t)
    return t


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    last_exc: json.JSONDecodeError | None = None
    data: Any = None
    for cand in candidates + [_repair_json(c) for c in candidates]:
        try:
            data = json.loads(cand)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
    else:
        raise last_exc or json.JSONDecodeError("no JSON object found", text, 0)
    if isinstance(data, list):  # some models return the bare cluster array
        data = {"clusters": data}
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
    return data


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
PROPER_NOUN_RX = re.compile(r"\b([A-Z][A-Za-z0-9&'.-]+(?:\s+(?:of|the|de|&)?\s*[A-Z][A-Za-z0-9&'.-]+){0,3})")
ENTITY_BLOCKLIST = frozenset({"The", "A", "An", "This", "That", "New", "Why", "How", "What", "Show", "Ask", "Launch", "HN"})


def extract_entities(texts: list[str], limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for t in texts:
        for m in PROPER_NOUN_RX.findall(t):
            m = m.strip(" .'-")
            words = m.split()
            if len(m) < 2 or m in ENTITY_BLOCKLIST or m.lower() in STOPWORDS:
                continue
            if words[0] in ENTITY_BLOCKLIST:
                if len(words) == 1:
                    continue
                m = " ".join(words[1:])
            if re.fullmatch(r"[A-Z]{1,4}s", m):  # 'TDs', 'QBs'
                continue
            counts[m] += 1
    return [e for e, _ in counts.most_common(limit)]


def clean_entities(entities: list[str], limit: int = 6) -> list[str]:
    """Keep proper-noun-like entities (must contain a capital or digit), ASCII, de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for e in entities:
        e = sanitize_headline(e, max_words=5)
        key = dedupe_key(e)
        if not e or not key or key in seen or key in STOPWORDS:
            continue
        if not re.search(r"[A-Z0-9]", e):
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= limit:
            break
    return out


def _item_text(it: CleanedTrendItem) -> str:
    """Title plus the context that actually names entities (Google Trends news headlines etc.)."""
    parts = [it.normalized_title]
    if it.source is SourceName.GOOGLE_TRENDS and it.metadata.get("news_titles"):
        parts.extend(str(t) for t in it.metadata["news_titles"][:3])
    elif it.source in (SourceName.REDDIT, SourceName.PRODUCTHUNT) and it.description:
        parts.append(it.description[:200])
    return normalize_text(" ".join(parts))


class LinkIndex:
    """Pairwise 'same specific topic' test used to validate LLM groupings.

    Two items are linked when they share distinctive tokens (not just a broad theme word) or
    when both literally name the same multi-character entity phrase.
    """

    def __init__(self, items: list[CleanedTrendItem], corpus: list[CleanedTrendItem] | None = None) -> None:
        # ``corpus`` = every cleaned item of the run (not just the LLM batch): more evidence for co-occurrence
        pool = {it.item_id: it for it in (corpus or [])}
        pool.update({it.item_id: it for it in items})
        self.text_keys = {iid: dedupe_key(_item_text(it)) for iid, it in pool.items()}
        self.toks = {iid: significant_tokens(_item_text(it)) for iid, it in pool.items()}
        self.df: Counter[str] = Counter(t for ts in self.toks.values() for t in ts)
        n = max(1, len(pool))
        self.rare_cap = max(4, int(0.06 * n))
        self._rx_cache: dict[str, re.Pattern[str] | None] = {}
        self._co_cache: dict[tuple[str, str], bool] = {}

    def _entity_rx(self, entity: str) -> re.Pattern[str] | None:
        key = dedupe_key(entity)
        if key not in self._rx_cache:
            self._rx_cache[key] = re.compile(r"\b" + re.escape(key) + r"\b") if len(key) >= 4 else None
        return self._rx_cache[key]

    def cooccur(self, e1: str, e2: str) -> bool:
        """True when some item of the run names both entities (e.g. a headline 'Packers vs. Falcons ...')."""
        k1, k2 = sorted((dedupe_key(e1), dedupe_key(e2)))
        if k1 == k2:
            return True
        if (k1, k2) not in self._co_cache:
            r1, r2 = self._entity_rx(e1), self._entity_rx(e2)
            self._co_cache[(k1, k2)] = bool(r1 and r2) and any(
                r1.search(t) and r2.search(t) for t in self.text_keys.values()
            )
        return self._co_cache[(k1, k2)]

    def linked(self, a: int, b: int) -> bool:
        ta, tb = self.toks.get(a, set()), self.toks.get(b, set())
        shared = ta & tb
        if not shared:
            return False
        rare = [t for t in shared if self.df[t] <= self.rare_cap]
        jacc = len(shared) / len(ta | tb)
        if len(shared) >= 2 and (jacc >= 0.25 or len(rare) >= 2):
            return True
        # short fragments ('Packers', 'Bijan') link to a fuller title via one distinctive name
        if rare and min(len(ta), len(tb)) <= 3 and any(len(t) >= 5 for t in rare):
            return True
        return False

    def mentions(self, item_id: int, entity: str) -> bool:
        rx = self._entity_rx(entity)
        return rx is not None and rx.search(self.text_keys.get(item_id, "")) is not None

    def components(self, ids: list[int], entities: list[str]) -> list[list[int]]:
        parent = {i: i for i in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if self.linked(ids[x], ids[y]):
                    parent[find(ids[y])] = find(ids[x])
        # entity evidence (phrase match only; no tail-token guessing):
        #  * two items naming the same entity are linked
        #  * items naming DIFFERENT entities are linked only when those entities co-occur in some
        #    other item of the run ('Packers' + 'Falcons' <- 'Packers vs. Falcons: ...'). An LLM
        #    entity list alone is never enough evidence ('Samsung' + 'F-Droid' never co-occur).
        holders = {e: [i for i in ids if self.mentions(i, e)] for e in entities}
        holders = {e: h for e, h in holders.items() if h}
        for h in holders.values():
            for x in h[1:]:
                parent[find(x)] = find(h[0])
        ents = list(holders)
        for a in range(len(ents)):
            for b in range(a + 1, len(ents)):
                if self.cooccur(ents[a], ents[b]):
                    parent[find(holders[ents[b]][0])] = find(holders[ents[a]][0])
        comps: dict[int, list[int]] = defaultdict(list)
        for i in ids:
            comps[find(i)].append(i)
        return sorted(comps.values(), key=len, reverse=True)


@dataclass
class DraftCluster:
    item_ids: list[int]
    headline: str
    category_raw: str
    entities: list[str] = field(default_factory=list)
    summary: str = ""
    relevance: int = 5
    needs_label: bool = False


@dataclass
class ClusterOutcome:
    """Result of stage 3. Every input item is in exactly one cluster or one discard bucket."""

    clusters: list[MacroCluster]
    discards: dict[str, list[int]]
    mode: str

    @property
    def discarded_count(self) -> int:
        return sum(len(v) for v in self.discards.values())

    def assert_partition(self, items: list[CleanedTrendItem]) -> None:
        """Enforce the partition invariant; repairs (and logs) any violation instead of crashing a run."""
        expected = {it.item_id for it in items}
        seen: set[int] = set()
        for c in self.clusters:
            unique = [i for i in dict.fromkeys(c.member_item_ids) if i in expected and i not in seen]
            if len(unique) != len(c.member_item_ids):
                log.error("accounting: cluster %s had duplicate/foreign members; repaired", c.cluster_id)
                by_id = {it.item_id: it for it in items}
                c.member_item_ids = unique
                c.raw_item_count = sum(by_id[i].raw_weight for i in unique) or 1
            seen.update(c.member_item_ids)
        for reason in list(self.discards):
            kept = [i for i in dict.fromkeys(self.discards[reason]) if i in expected and i not in seen]
            if len(kept) != len(self.discards[reason]):
                log.error("accounting: discard bucket %s overlapped clusters/other buckets; repaired", reason)
            self.discards[reason] = kept
            seen.update(kept)
        missing = expected - seen
        if missing:
            log.error("accounting: %d items were neither clustered nor discarded; recorded as unsupported_grouping", len(missing))
            self.discards.setdefault("unsupported_grouping", []).extend(sorted(missing))
        self.discards = {k: v for k, v in self.discards.items() if v}


# ======================================================================= clusterer
class SemanticClusterer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._use_schema = True

    @property
    def batch_size(self) -> int:
        return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, self.settings.llm_batch_size))

    # ....................................................... Ollama plumbing
    def _get_client(self):
        if self._client is None:
            from ollama import AsyncClient  # imported lazily so heuristic mode works without the package

            self._client = AsyncClient(host=self.settings.ollama_host, timeout=self.settings.ollama_timeout_s)
        return self._client

    async def health_check(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
        except ImportError:
            return False, "python package 'ollama' not installed"
        try:
            listing = await client.list()
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama unreachable at {self.settings.ollama_host} ({type(exc).__name__})"
        models = getattr(listing, "models", None)
        if models is None and isinstance(listing, dict):
            models = listing.get("models", [])
        names: set[str] = set()
        for m in models or []:
            name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
            if name:
                names.add(str(name))
        wanted = self.settings.ollama_model
        if wanted in names or f"{wanted}:latest" in names or any(n.split(":")[0] == wanted for n in names):
            return True, "ok"
        return False, f"model '{wanted}' not pulled (run: ollama pull {wanted})"

    async def _chat_json(
        self, system: str, user: str, schema: dict[str, Any], label: str = "LLM call"
    ) -> dict[str, Any]:
        """One structured Ollama call with live progress logging.

        * logs a heartbeat every 60 s so a slow CPU inference never looks like a hang
        * logs duration and generation speed (tokens/s) when the call returns
        * a timeout is NOT retried: re-sending the same prompt to an overloaded CPU would only
          burn another full timeout window; the caller falls back to heuristics instead
        """
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            started = time.perf_counter()
            heartbeat = asyncio.create_task(self._heartbeat(label, started))
            try:
                resp = await client.chat(
                    model=self.settings.ollama_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    format=schema if self._use_schema else "json",
                    options={
                        "temperature": self.settings.ollama_temperature,
                        "num_ctx": self.settings.ollama_num_ctx,
                        "num_predict": 2048,
                    },
                    keep_alive=self.settings.ollama_keep_alive,
                )
                self._log_call_stats(label, resp, time.perf_counter() - started)
                message = getattr(resp, "message", None)
                content = getattr(message, "content", None) if message is not None else None
                if content is None and isinstance(resp, dict):
                    content = resp.get("message", {}).get("content")
                return extract_json(content or "")
            except json.JSONDecodeError as exc:
                last_exc = exc
                log.warning("%s: invalid JSON (attempt %d): %s", label, attempt + 1, exc)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc).lower()
                if self._use_schema and ("format" in msg or "schema" in msg):
                    log.info("server rejected JSON-schema format; falling back to format='json'")
                    self._use_schema = False
                    continue
                elapsed = time.perf_counter() - started
                if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)) or "timed out" in msg:
                    log.warning("%s: timed out after %.0f s - not retrying (falls back to heuristics)", label, elapsed)
                    raise ClusteringError(f"{label} timed out after {elapsed:.0f}s") from exc
                log.warning("%s failed (attempt %d): %s: %s", label, attempt + 1, type(exc).__name__, exc)
            finally:
                heartbeat.cancel()
        raise ClusteringError(f"{label} failed after retries: {last_exc}")

    @staticmethod
    async def _heartbeat(label: str, started: float, every: float = 60.0) -> None:
        try:
            while True:
                await asyncio.sleep(every)
                log.info("%s: still generating (%.0f s elapsed)", label, time.perf_counter() - started)
        except asyncio.CancelledError:
            return

    @staticmethod
    def _log_call_stats(label: str, resp: Any, elapsed: float) -> None:
        def field(name: str) -> Any:
            v = getattr(resp, name, None)
            if v is None and isinstance(resp, dict):
                v = resp.get(name)
            return v

        prompt_tokens, gen_tokens, gen_ns = field("prompt_eval_count"), field("eval_count"), field("eval_duration")
        if gen_tokens and gen_ns:
            log.info(
                "%s: done in %.0f s (%s prompt tokens, %s generated @ %.1f tok/s)",
                label, elapsed, prompt_tokens or "?", gen_tokens, gen_tokens / (gen_ns / 1e9),
            )
        else:
            log.info("%s: done in %.0f s", label, elapsed)

    # ....................................................... public entry point
    async def cluster(
        self,
        items: list[CleanedTrendItem],
        corpus: list[CleanedTrendItem] | None = None,
        use_llm: bool = True,
    ) -> ClusterOutcome:
        """Group, label, isolate and validate ``items``.

        ``corpus`` = every cleaned item of the run (extra co-occurrence evidence for entity isolation).
        ``use_llm=False`` skips Ollama entirely (no embeddings, heuristic labels).
        """
        if not items:
            return ClusterOutcome(clusters=[], discards={}, mode="empty")
        by_id = {it.item_id: it for it in items}
        index = LinkIndex(items, corpus)
        discards: dict[str, list[int]] = defaultdict(list)

        llm_ok, reason = (await self.health_check()) if use_llm else (False, "--no-llm")
        if use_llm and not llm_ok:
            log.warning("LLM unavailable (%s) -> heuristic labels", reason)

        # ---- 3a density grouping
        started = time.perf_counter()
        groups, noise, method = await self._group(items, try_embeddings=use_llm)
        if self.settings.outlier_policy == "keep_top":
            keep = [i for i in noise if by_id[i].heuristic_score >= self.settings.singleton_keep_score]
            groups.extend([[i] for i in keep])
            noise = [i for i in noise if i not in set(keep)]
        discards["density_noise"].extend(noise)
        drafts = [DraftCluster(item_ids=list(g), headline="", category_raw="", needs_label=True) for g in groups]
        log.info("stage 3a: %d groups, %d noise items via %s in %.1f s", len(drafts), len(noise), method, time.perf_counter() - started)

        # ---- 3b label, 3c isolate (+ re-label split parts, offer strays back to them)
        if llm_ok:
            drafts, _ = await self._relabel(drafts, [], by_id)
            drafts, orphans = self._enforce_coherence(drafts, index)
            drafts, orphans = await self._relabel(drafts, orphans, by_id)
            mode = f"ollama:{self.settings.ollama_model} + {method}"
        else:
            self._heuristic_labels(drafts, by_id)
            drafts, orphans = self._enforce_coherence(drafts, index)
            self._heuristic_labels(drafts, by_id)
            mode = f"heuristic ({reason}) + {method}"
        orphans = self._rehome_orphans(drafts, orphans, index)
        discards["unsupported_grouping"].extend(orphans)

        # ---- 3d merge, 3e finalise
        drafts = self._deterministic_merge(drafts, index)
        clusters, final_discards = self._finalize(drafts, by_id)
        for r, ids in final_discards.items():
            discards[r].extend(ids)

        outcome = ClusterOutcome(clusters=clusters, discards={k: v for k, v in discards.items() if v}, mode=mode)
        outcome.assert_partition(items)
        return outcome

    async def _group(self, items: list[CleanedTrendItem], try_embeddings: bool) -> tuple[list[list[int]], list[int], str]:
        if try_embeddings:
            try:
                client = self._get_client()
                vectors = await embed_items(client, self.settings, items)
                res = density_cluster(items, vectors, self.settings)
                return res.groups, res.noise, res.method
            except ImportError:
                log.warning("ollama package missing -> lexical grouping")
            except EmbeddingUnavailable as exc:
                log.warning(
                    "embeddings unavailable (%s) -> lexical grouping. Fix: ollama pull %s",
                    str(exc)[:160], self.settings.embed_model,
                )
        groups = self._lexical_groups(items)
        clusters = [g for g in groups if len(g) >= 2]
        noise = [g[0] for g in groups if len(g) == 1]
        return clusters, noise, "lexical (no embeddings)"

    def _heuristic_labels(self, drafts: list[DraftCluster], by_id: dict[int, CleanedTrendItem]) -> None:
        for d in drafts:
            if d.needs_label or not d.headline:
                h = self._heuristic_draft([by_id[i] for i in d.item_ids])
                d.headline, d.category_raw, d.summary, d.entities, d.relevance = (
                    h.headline, h.category_raw, h.summary, h.entities, h.relevance,
                )
                d.needs_label = False

    @staticmethod
    def _render_item(it: CleanedTrendItem) -> str:
        src = it.source.value
        if it.source is SourceName.REDDIT and it.metadata.get("subreddit"):
            src = f"reddit/r/{it.metadata['subreddit']}"
        hint = it.category_hint.value if it.category_hint else "-"
        ctx = ""
        if it.context:
            ctx = normalize_text(it.context)[:260]
        elif it.source is SourceName.GOOGLE_TRENDS and it.metadata.get("news_titles"):
            ctx = normalize_text(" / ".join(it.metadata["news_titles"][:2]))[:160]
        elif it.description:
            ctx = normalize_text(it.description)[:160]
        return f"{src} | {hint} | {it.normalized_title[:200]} | " + (ctx if ctx else "(no context)")

    # ....................................................... coherence
    def _enforce_coherence(
        self, drafts: list[DraftCluster], index: LinkIndex
    ) -> tuple[list[DraftCluster], list[int]]:
        """Split clusters whose members do not form one connected same-topic graph.

        * one coherent core (+ stray items): keep the core with its label; strays that literally
          name one of the cluster's entities stay (entity resolution, e.g. 'Jordan Love' in a
          Packers cluster), the rest become orphans for re-assignment.
        * two or more cores: a Frankenstein cluster. Every core becomes its own draft that must
          be re-labelled; strays follow the core that owns the entity they mention.
        """
        out: list[DraftCluster] = []
        orphans: list[int] = []
        split_count = 0
        for d in drafts:
            if len(d.item_ids) <= 1:
                out.append(d)
                continue
            comps = index.components(d.item_ids, d.entities)
            if len(comps) == 1:
                out.append(d)
                continue
            cores = [c for c in comps if len(c) >= 2]
            strays = [c[0] for c in comps if len(c) == 1]

            if not cores:
                # no two items share a topic by tokens or co-occurring entities: the grouping is unsupported
                orphans.extend(d.item_ids)
                split_count += 1
                continue

            if len(cores) == 1:
                # strays already had every chance to link (tokens, shared or co-occurring entities)
                core = list(cores[0])
                orphans.extend(strays)
                ejected = len(strays)
                out.append(
                    DraftCluster(
                        item_ids=core,
                        headline=d.headline,
                        category_raw=d.category_raw,
                        entities=d.entities,
                        summary=d.summary,
                        relevance=d.relevance,
                        # a label written for a much larger mixed set is no longer trustworthy
                        needs_label=ejected * 3 > len(d.item_ids),
                    )
                )
                continue

            split_count += 1
            out.extend(DraftCluster(item_ids=list(c), headline="", category_raw="", needs_label=True) for c in cores)
            orphans.extend(strays)
        for d in out:
            grounded = self._ground_entities(d.entities, d.item_ids, index)
            # a label whose entities mostly don't appear in its own items, or an umbrella
            # headline, was written for a different/mixed set of signals: re-label it
            if d.entities and len(grounded) * 2 < len(d.entities):
                d.needs_label = True
            if d.headline and is_generic_headline(sanitize_headline(d.headline, HEADLINE_MAX_WORDS)):
                d.needs_label = True
            d.entities = grounded
        if split_count or orphans:
            log.info("coherence: split %d mixed clusters, %d items orphaned for re-assignment", split_count, len(orphans))
        return out, orphans

    @staticmethod
    def _rehome_orphans(drafts: list[DraftCluster], orphans: list[int], index: LinkIndex) -> list[int]:
        """Attach an orphan to the ONE cluster whose (grounded) entities it literally names.

        Entities were already grounded in each cluster's own items, so a literal match is real
        evidence ('Falcons' -> the Packers vs. Falcons cluster). Ambiguous matches stay orphaned.
        """
        remaining: list[int] = []
        attached = 0
        for o in orphans:
            homes = [d for d in drafts if any(index.mentions(o, e) for e in d.entities)]
            if len(homes) == 1:
                homes[0].item_ids.append(o)
                attached += 1
            else:
                remaining.append(o)
        if orphans:
            log.info("stage 3c: %d/%d orphans re-homed by literal entity match", attached, len(orphans))
        return remaining

    @staticmethod
    def _ground_entities(entities: list[str], item_ids: list[int], index: LinkIndex) -> list[str]:
        """Drop entities that no member of the cluster actually names (leftovers of a mixed label)."""
        member_tokens: set[str] = set()
        for i in item_ids:
            member_tokens |= index.toks.get(i, set())
        kept = []
        for e in entities:
            if any(index.mentions(i, e) for i in item_ids):
                kept.append(e)
                continue
            etoks = {t for t in significant_tokens(e) if len(t) >= 4}
            if etoks and etoks <= member_tokens:
                kept.append(e)
        return kept

    async def _relabel(
        self, drafts: list[DraftCluster], orphans: list[int], by_id: dict[int, CleanedTrendItem]
    ) -> tuple[list[DraftCluster], list[int]]:
        """LLM labels re-grouped drafts and may re-home orphans into them (never regroups)."""
        pending = [d for d in drafts if d.needs_label]
        if not pending and not orphans:
            return drafts, []
        if not pending:
            return drafts, orphans

        remaining_orphans = list(orphans)
        chunk: list[DraftCluster] = []
        chunks: list[list[DraftCluster]] = []
        for d in sorted(pending, key=lambda x: -len(x.item_ids)):
            if chunk and sum(len(c.item_ids) for c in chunk) + len(d.item_ids) > self.batch_size:
                chunks.append(chunk)
                chunk = []
            chunk.append(d)
        if chunk:
            chunks.append(chunk)

        for ci, group_chunk in enumerate(chunks, start=1):
            room = max(0, self.batch_size - sum(len(d.item_ids) for d in group_chunk))
            offered = remaining_orphans[: max(room, 5)]
            lines: list[str] = []
            for gi, d in enumerate(group_chunk, start=1):
                lines.append(f"Group {gi}:")
                for m in d.item_ids[:10]:
                    lines.append(f"  - {self._render_item(by_id[m])}")
            if offered:
                lines.append("")
                lines.append("Unassigned signals (id | source | hint | title | context):")
                for oi, o in enumerate(offered, start=1):
                    lines.append(f"{oi} | {self._render_item(by_id[o])}")
            lines.append("")
            lines.append(f"Label groups 1..{len(group_chunk)}" + (" and assign each unassigned signal." if offered else "."))
            log.info("LLM label %d/%d (%d groups, %d items, %d unassigned)", ci, len(chunks), len(group_chunk), sum(len(d.item_ids) for d in group_chunk), len(offered))
            try:
                data = await self._chat_json(CLUSTER_SYSTEM_PROMPT, "\n".join(lines), _relabel_schema(), f"label {ci}/{len(chunks)}")
                parsed = _RelabelResponse.model_validate(data)
            except (ClusteringError, ValueError) as exc:
                log.warning("relabel pass failed (%s); using heuristic labels", exc)
                parsed = _RelabelResponse()
            labels = {g.group_id: g for g in parsed.groups}
            for gi, d in enumerate(group_chunk, start=1):
                g = labels.get(gi)
                if g and normalize_text(g.headline):
                    d.headline, d.category_raw, d.summary = g.headline, g.category, g.summary
                    d.entities = [normalize_text(e) for e in g.primary_entities if normalize_text(e)][:6]
                    d.relevance = g.relevance_score
                else:
                    h = self._heuristic_draft([by_id[i] for i in d.item_ids])
                    d.headline, d.category_raw, d.summary, d.entities, d.relevance = (
                        h.headline, h.category_raw, h.summary, h.entities, h.relevance,
                    )
                d.needs_label = False
            for a in parsed.assignments:
                if 1 <= a.item_id <= len(offered) and 1 <= a.group_id <= len(group_chunk):
                    oid = offered[a.item_id - 1]
                    if oid in remaining_orphans:
                        group_chunk[a.group_id - 1].item_ids.append(oid)
                        remaining_orphans.remove(oid)
        return drafts, remaining_orphans

    # ....................................................... merging
    @staticmethod
    def _entity_keys(d: DraftCluster) -> set[str]:
        return {dedupe_key(e) for e in d.entities if len(dedupe_key(e)) >= 3}

    def _merge_drafts(self, group: list[DraftCluster]) -> DraftCluster:
        group = sorted(group, key=lambda d: (len(d.item_ids), d.relevance), reverse=True)
        lead = group[0]
        ids: list[int] = []
        entities: list[str] = []
        seen_e: set[str] = set()
        cat_votes: Counter[str] = Counter()
        for d in group:
            ids.extend(i for i in d.item_ids if i not in ids)
            cat_votes[coerce_category(d.category_raw).value] += len(d.item_ids)
            for e in d.entities:
                k = dedupe_key(e)
                if k and k not in seen_e:
                    seen_e.add(k)
                    entities.append(e)
        return DraftCluster(
            item_ids=ids,
            headline=lead.headline,
            category_raw=cat_votes.most_common(1)[0][0],
            entities=entities[:6],
            summary=lead.summary,
            relevance=max(d.relevance for d in group),
        )

    def _deterministic_merge(self, drafts: list[DraftCluster], index: LinkIndex | None = None) -> list[DraftCluster]:
        changed = True
        while changed and len(drafts) > 1:
            changed = False
            for i in range(len(drafts)):
                for j in range(i + 1, len(drafts)):
                    a, b = drafts[i], drafts[j]
                    ka, kb = self._entity_keys(a), self._entity_keys(b)
                    same_headline = bool(dedupe_key(a.headline)) and dedupe_key(a.headline) == dedupe_key(b.headline)
                    jacc = len(ka & kb) / len(ka | kb) if ka and kb else 0.0
                    subset = len(ka & kb) >= 2 and (ka <= kb or kb <= ka)
                    if same_headline or jacc >= 0.5 or subset:
                        if index is not None and not any(index.linked(x, y) for x in a.item_ids for y in b.item_ids):
                            identical_entities = bool(ka) and ka == kb
                            if not (identical_entities or (jacc >= 0.5 and len(ka & kb) >= 2)):
                                continue
                        merged = self._merge_drafts([a, b])
                        drafts = [d for k, d in enumerate(drafts) if k not in (i, j)] + [merged]
                        changed = True
                        break
                if changed:
                    break
        return drafts

    # ....................................................... validation
    def _guard_category(self, proposed: CategoryEnum, members: list[CleanedTrendItem]) -> CategoryEnum:
        hints = Counter(m.category_hint for m in members if m.category_hint)
        votes: Counter[CategoryEnum] = Counter()
        for m in members:
            votes.update(category_votes(f"{m.normalized_title} {m.description or ''}"))
        sources = {m.source for m in members}

        if sources == {SourceName.ARXIV}:
            return CategoryEnum.SCIENCE_AI
        if proposed in (CategoryEnum.TECH, CategoryEnum.SCIENCE_AI):
            tech_hint = hints[CategoryEnum.TECH] + hints[CategoryEnum.SCIENCE_AI]
            tech_votes = votes[CategoryEnum.TECH] + votes[CategoryEnum.SCIENCE_AI]
            other = [(c, n) for c, n in votes.most_common() if c not in (CategoryEnum.TECH, CategoryEnum.SCIENCE_AI)]
            if tech_hint == 0 and (tech_votes == 0 or (other and other[0][1] >= 2 * tech_votes)):
                if other:
                    return other[0][0]
                non_tech_hints = [c for c, _ in hints.most_common() if c not in (CategoryEnum.TECH, CategoryEnum.SCIENCE_AI)]
                return non_tech_hints[0] if non_tech_hints else CategoryEnum.NEWS
        if proposed is CategoryEnum.SPORTS and votes[CategoryEnum.SPORTS] == 0 and hints[CategoryEnum.SPORTS] == 0:
            top = votes.most_common(1)
            if top and top[0][1] >= 2:
                return top[0][0]
        return proposed

    @staticmethod
    def _best_member_title(members: list[CleanedTrendItem]) -> str:
        """Most descriptive strong title: prefers multi-word headlines over bare hashtags."""

        def rank(m: CleanedTrendItem) -> float:
            n_tok = len(significant_tokens(m.normalized_title))
            return m.heuristic_score + 0.04 * min(n_tok, 8) - (0.3 if n_tok < 2 else 0.0)

        ordered = sorted(members, key=lambda m: m.heuristic_score, reverse=True)
        title = max(ordered[:6], key=rank).normalized_title
        return title.title() if title.islower() else title

    def _fix_summary(
        self, summary: str, members: list[CleanedTrendItem], headline: str, entities: list[str] | None = None
    ) -> str:
        text = sanitize_summary(summary, entities)
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 3] if text else []
        sources = sorted({m.source.value for m in members})
        if len(sources) > 1:
            closing = f"The story is trending across {display_sources(sources)} at the same time."
        else:
            closing = f"{display_sources(sources)} is currently driving attention to the story."
        if not sentences:
            sentences = [f"{headline.rstrip('.')} is drawing attention across trend sources."]
        if len(sentences) == 1:
            sentences.append(closing)
        return sanitize_summary(" ".join(sentences[:2]), entities)

    @staticmethod
    def is_insufficient(summary: str, headline: str = "") -> bool:
        """True for the explicit [INSUFFICIENT_DATA] flag or placeholder/filler summaries."""
        text = f"{headline} {summary}"
        return bool(INSUFFICIENT_RX.search(text) or FILLER_RX.search(summary or "") or not normalize_text(summary or ""))

    def _finalize(
        self, drafts: list[DraftCluster], by_id: dict[int, CleanedTrendItem]
    ) -> tuple[list[MacroCluster], dict[str, list[int]]]:
        clusters: list[MacroCluster] = []
        discards: dict[str, list[int]] = defaultdict(list)
        s = self.settings
        for d in drafts:
            members = [by_id[i] for i in dict.fromkeys(d.item_ids) if i in by_id]
            if not members:
                continue
            ids = [m.item_id for m in members]
            if self.is_insufficient(d.summary, d.headline):
                discards["insufficient_data"].extend(ids)
                continue
            if d.relevance < s.min_cluster_relevance:  # i.e. relevance <= 3 with the default floor of 4
                discards["low_relevance"].extend(ids)
                continue
            best_score = max(m.heuristic_score for m in members)
            multi_source = len({m.source for m in members}) > 1 or any(m.duplicate_count > 1 for m in members)
            if len(members) < s.min_cluster_items and not multi_source:
                if not (best_score >= s.singleton_keep_score or d.relevance >= s.singleton_keep_relevance):
                    discards["weak_singleton"].extend(ids)
                    continue
            fallback = Counter(
                m.category_hint or m.inferred_category or CategoryEnum.NEWS for m in members
            ).most_common(1)[0][0]
            category = self._guard_category(coerce_category(d.category_raw, fallback), members)

            headline = sanitize_headline(d.headline, HEADLINE_MAX_WORDS)
            if not headline or is_generic_headline(headline):
                headline = sanitize_headline(self._best_member_title(members), HEADLINE_MAX_WORDS)

            entities = clean_entities(d.entities)
            if not entities:
                entities = clean_entities(extract_entities([m.normalized_title for m in members]))
            urls: list[str] = []
            for m in sorted(members, key=lambda x: x.heuristic_score, reverse=True):
                for u in [m.url, *m.merged_urls]:
                    if u and u not in urls:
                        urls.append(u)
            clusters.append(
                MacroCluster(
                    cluster_id=MacroCluster.make_id(headline, entities),
                    headline=headline,
                    category=category,
                    relevance_score=max(1, min(10, d.relevance)),
                    llm_relevance=d.relevance,
                    velocity_score=50.0,
                    summary=self._fix_summary(d.summary, members, headline, entities),
                    primary_entities=entities,
                    source_urls=urls[:10],
                    raw_item_count=sum(m.raw_weight for m in members),
                    sources=sorted({m.source.value for m in members}),
                    member_item_ids=[m.item_id for m in members],
                )
            )
        # identical ids can arise when two clusters resolve to the same entity set -> merge by id
        unique: dict[str, MacroCluster] = {}
        for c in clusters:
            if c.cluster_id in unique:
                prev = unique[c.cluster_id]
                prev.member_item_ids.extend(c.member_item_ids)
                prev.raw_item_count += c.raw_item_count
                prev.source_urls = list(dict.fromkeys(prev.source_urls + c.source_urls))[:10]
                prev.sources = sorted(set(prev.sources) | set(c.sources))
                prev.relevance_score = max(prev.relevance_score, c.relevance_score)
            else:
                unique[c.cluster_id] = c
        return list(unique.values()), dict(discards)

    # ....................................................... heuristic path
    def _lexical_groups(self, items: list[CleanedTrendItem]) -> list[list[int]]:
        """Union-find over distinctive shared tokens (inverted index, ~O(n * k))."""
        toks = {it.item_id: significant_tokens(it.normalized_title) for it in items}
        df: Counter[str] = Counter(t for ts in toks.values() for t in ts)
        n = max(1, len(items))
        rare_cap = max(4, int(0.06 * n))
        short_cap = max(6, int(0.10 * n))
        parent = {it.item_id: it.item_id for it in items}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        index: dict[str, list[int]] = defaultdict(list)
        for iid, ts in toks.items():
            for t in ts:
                if df[t] <= max(short_cap, 12):
                    index[t].append(iid)
        checked: set[tuple[int, int]] = set()
        for t, ids in index.items():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    a, b = ids[x], ids[y]
                    key = (a, b) if a < b else (b, a)
                    if key in checked:
                        continue
                    checked.add(key)
                    ta, tb = toks[a], toks[b]
                    shared = ta & tb
                    rare_shared = {s for s in shared if df[s] <= rare_cap}
                    jacc = len(shared) / len(ta | tb) if ta | tb else 0.0
                    short = min(len(ta), len(tb)) <= 2
                    short_shared = short and any(df[s] <= short_cap for s in shared)
                    if jacc >= 0.34 or len(rare_shared) >= 2 or (rare_shared and short) or short_shared:
                        union(a, b)
        groups: dict[int, list[int]] = defaultdict(list)
        for it in items:
            groups[find(it.item_id)].append(it.item_id)
        return list(groups.values())

    def _heuristic_draft(self, members: list[CleanedTrendItem]) -> DraftCluster:
        members = sorted(members, key=lambda m: m.heuristic_score, reverse=True)
        lead = members[0]
        entities = extract_entities([m.normalized_title for m in members])
        votes: Counter[CategoryEnum] = Counter()
        for m in members:
            votes.update(category_votes(m.normalized_title))
            if m.category_hint:
                votes[m.category_hint] += 2
        category = votes.most_common(1)[0][0] if votes else CategoryEnum.NEWS
        sources = sorted({m.source.value for m in members})
        subject = ", ".join(entities[:3]) or lead.normalized_title[:80]
        n = sum(m.raw_weight for m in members)
        with_ctx = next((m for m in members if m.context), None)
        if with_ctx is not None:
            # heuristic mode still reports facts: the lead sentence of the best scraped context
            first = SENTENCE_SPLIT.split(sanitize_summary(with_ctx.context.split(" | ")[-1]))[0]
            summary = (
                f"{first.rstrip('.')}. "
                f"{display_sources(sources)} {'are' if len(sources) > 1 else 'is'} carrying "
                f"{n} related signal{'s' if n != 1 else ''} about {subject}."
            )
        else:
            summary = INSUFFICIENT_FLAG  # nothing factual to say -> dropped in _finalize
        mean = sum(m.heuristic_score for m in members) / len(members)
        relevance = max(1, min(10, round(1 + 9 * (0.6 * mean + 0.4 * min(1.0, len(sources) / 3)))))
        return DraftCluster(
            item_ids=[m.item_id for m in members],
            headline=self._best_member_title(members)[:100],
            category_raw=category.value,
            entities=entities,
            summary=summary,
            relevance=relevance,
        )
