"""Central configuration for Agent Reach.

Every value can be overridden with an environment variable prefixed with
``AGENT_REACH_`` (e.g. ``AGENT_REACH_OLLAMA_HOST=http://10.0.0.5:11434``) or via a
``.env`` file in the working directory. List values accept JSON
(e.g. ``AGENT_REACH_REDDIT_SUBREDDITS='["popular","nfl"]'``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_REACH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_s: float = 240.0
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.1
    ollama_keep_alive: str = "10m"
    llm_batch_size: int = Field(default=20, ge=5, le=25)  # >25 makes llama3.1:8b emit malformed JSON
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_enable_merge_pass: bool = True

    # -------------------------------------------------------------- storage
    db_path: Path = Path("agent_reach.db")
    retention_days: int = Field(default=30, ge=1)

    # ----------------------------------------------------------------- HTTP
    http_timeout_s: float = 10.0
    http_max_retries: int = Field(default=3, ge=0, le=8)
    http_backoff_base_s: float = 1.0
    http_backoff_max_s: float = 16.0
    http_max_concurrency: int = Field(default=8, ge=1, le=32)
    contact_email: str = "agent-reach@example.invalid"  # Wikimedia/Reddit ask for a contact in the UA

    # -------------------------------------------------------------- sources
    geo: str = "US"
    trends24_region: str = "united-states"
    wikipedia_project: str = "en.wikipedia"
    reddit_subreddits: list[str] = Field(
        default_factory=lambda: ["popular", "news", "worldnews", "technology", "science", "sports", "movies"]
    )
    arxiv_categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "cs.CV"])
    max_items_per_source: int = Field(default=40, ge=5, le=200)
    enabled_sources: list[str] = Field(
        default_factory=lambda: [
            "x_trends24",
            "reddit",
            "tiktok",
            "google_trends",
            "google_news",
            "wikipedia",
            "arxiv",
            "hackernews",
            "github",
            "producthunt",
        ]
    )

    # ------------------------------------------------------ noise thresholds
    reddit_min_score: int = 20
    reddit_min_comments: int = 5
    reddit_allow_unverified_rss: bool = False  # keep ALL RSS items regardless of rank
    reddit_rss_max_rank: int = Field(default=10, ge=0, le=100)  # RSS "top of day" ranks kept without metrics
    reddit_timeout_s: float = 3.0  # per-endpoint timeout so blocked JSON fails fast to RSS
    reddit_concurrency: int = Field(default=3, ge=1, le=6)
    reddit_rss_spacing_s: float = Field(default=2.0, ge=0.0, le=10.0)  # gap between RSS calls once JSON is blocked
    hn_min_points: int = 10
    github_min_stars_today: int = 20
    min_title_chars: int = 3
    max_items_for_llm: int = Field(default=150, ge=10, le=600)
    min_items_per_source_for_llm: int = Field(default=6, ge=0)

    # ----------------------------------------------------------- clustering
    min_cluster_items: int = Field(default=2, ge=1)
    singleton_keep_score: float = Field(default=0.80, ge=0.0, le=1.0)
    singleton_keep_relevance: int = Field(default=7, ge=1, le=10)
    min_cluster_relevance: int = Field(default=4, ge=1, le=10)  # LLM relevance below this is dropped

    # -------------------------------------------------------------- scoring
    velocity_windows_hours: list[float] = Field(default_factory=lambda: [1.0, 6.0, 24.0])
    velocity_window_weights: list[float] = Field(default_factory=lambda: [0.5, 0.3, 0.2])
    rank_relevance_weight: float = Field(default=0.6, ge=0.0, le=1.0)

    # --------------------------------------------------------------- report
    report_top_n: int = Field(default=15, ge=1, le=100)
    report_width: int = Field(default=100, ge=70, le=200)

    @field_validator("velocity_window_weights")
    @classmethod
    def _weights_positive(cls, v: list[float]) -> list[float]:
        if any(w < 0 for w in v):
            raise ValueError("velocity_window_weights must be non-negative")
        return v

    @field_validator("ollama_host")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
