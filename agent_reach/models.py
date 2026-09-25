"""Pydantic V2 data contracts shared by every Agent Reach stage."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_ids(v: Any) -> list[int]:
    """Accept ints, numeric strings, '#12', '[12]' -- LLMs are inconsistent."""
    out: list[int] = []
    if v is None:
        return out
    if not isinstance(v, (list, tuple, set)):
        v = [v]
    for x in v:
        try:
            out.append(int(str(x).strip().lstrip("#[").rstrip("]")))
        except ValueError:
            continue
    return out


class CategoryEnum(str, Enum):
    SPORTS = "Sports"
    ENTERTAINMENT = "Entertainment"
    TECH = "Tech"
    NEWS = "News"
    INTERNET_CULTURE = "Internet Culture"
    SCIENCE_AI = "Science & AI"

    @classmethod
    def values(cls) -> list[str]:
        return [c.value for c in cls]


class SourceName(str, Enum):
    X_TRENDS24 = "x_trends24"
    REDDIT = "reddit"
    TIKTOK = "tiktok"
    GOOGLE_TRENDS = "google_trends"
    GOOGLE_NEWS = "google_news"
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    HACKERNEWS = "hackernews"
    GITHUB = "github"
    PRODUCTHUNT = "producthunt"


class RawTrendItem(BaseModel):
    """A single raw signal as produced by an ingester."""

    model_config = ConfigDict(use_enum_values=False)

    title: str = Field(min_length=1, max_length=1000)
    source: SourceName
    category_hint: CategoryEnum | None = None
    raw_score: float | None = None
    url: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    comment_count: int | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    @property
    def content_hash(self) -> str:
        basis = f"{self.source.value}|{self.title.strip().lower()}|{self.url or ''}"
        return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()


class CleanedTrendItem(RawTrendItem):
    """A raw item that survived heuristic filtering, enriched for clustering."""

    item_id: int
    normalized_title: str
    heuristic_score: float = Field(ge=0.0, le=1.0)
    duplicate_count: int = 1
    merged_urls: list[str] = Field(default_factory=list)
    inferred_category: CategoryEnum | None = None


class VelocityWindow(BaseModel):
    window_hours: float
    baseline_rate: float | None = None
    current_rate: float
    growth: float | None = None


class MacroCluster(BaseModel):
    cluster_id: str
    headline: str
    category: CategoryEnum
    relevance_score: int = Field(ge=1, le=10)
    velocity_score: float = Field(ge=0.0, le=100.0)
    summary: str
    primary_entities: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    raw_item_count: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    # Enrichment (populated by clusterer/scorer)
    sources: list[str] = Field(default_factory=list)
    llm_relevance: int | None = None
    momentum: str = "STEADY"
    velocity_basis: str = "cold_start"
    velocity_windows: list[VelocityWindow] = Field(default_factory=list)
    combined_score: float = 0.0
    member_item_ids: list[int] = Field(default_factory=list, exclude=True)

    @staticmethod
    def make_id(headline: str, entities: list[str]) -> str:
        keys = sorted({e.strip().lower() for e in entities if e.strip()}) or [headline.strip().lower()]
        return hashlib.sha1("|".join(keys).encode("utf-8", "ignore")).hexdigest()[:12]


class SourceStat(BaseModel):
    source: str
    ok: bool
    item_count: int
    latency_ms: int
    error: str | None = None


class PipelineReport(BaseModel):
    run_id: str
    execution_time: float = Field(description="Wall-clock seconds for the full run")
    ingested_count: int
    filtered_count: int = Field(description="Items that survived heuristic filtering")
    cluster_count: int
    macro_clusters: list[MacroCluster]
    started_at: datetime = Field(default_factory=utcnow)
    llm_mode: str = "ollama"
    discarded_by_llm: int = 0
    source_stats: list[SourceStat] = Field(default_factory=list)
    filter_breakdown: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------- LLM I/O
class LLMCluster(BaseModel):
    """Loose schema for what the LLM returns (validated/coerced afterwards)."""

    headline: str
    category: str
    item_ids: list[int]
    primary_entities: list[str] = Field(default_factory=list)
    summary: str = ""
    relevance_score: int = 5

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> int:
        try:
            return max(1, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            return 5

    @field_validator("item_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[int]:
        return _coerce_ids(v)


class LLMClusterResponse(BaseModel):
    clusters: list[LLMCluster] = Field(default_factory=list)
    discarded_item_ids: list[int] = Field(default_factory=list)

    @field_validator("discarded_item_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[int]:
        return _coerce_ids(v)


class LLMMergeGroup(BaseModel):
    cluster_ids: list[int]
    headline: str = ""

    @field_validator("cluster_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[int]:
        return _coerce_ids(v)


class LLMMergeResponse(BaseModel):
    groups: list[LLMMergeGroup] = Field(default_factory=list)
