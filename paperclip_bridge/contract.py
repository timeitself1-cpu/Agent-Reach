"""File contract between the bridge and the research CLI.

The research CLI is invoked as ``<cmd> --in <request.json> --out <result.json>``.
It reads a ``ResearchRequest`` and must write a ``ResearchResult``. Exit codes:

* ``0``: ``result.json`` was written. Research failures are reported inside the
  file (``status="error"``), not through the exit code.
* anything else: the CLI crashed; the bridge marks the ticket ``blocked``.

Both files carry ``schema_version``. Changing or removing fields requires a
version bump; adding optional fields does not. The contract holds no Paperclip
identifiers except the opaque ``request_id`` the CLI may log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class ResearchRequest(BaseModel):
    """Input file: one research task, written by the bridge."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    request_id: str = Field(description="Opaque id for logs and output file names.")
    title: str = Field(description="One-line task, e.g. 'Summarise this week\\'s agent papers'.")
    instructions: str = Field(default="", description="Free-text task detail (the ticket description).")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperSynthesis(BaseModel):
    """One synthesised paper."""

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str = ""
    arxiv_id: str = ""
    summary: str
    key_claims: list[str] = Field(default_factory=list)
    methods: str = ""
    implementation_notes: str = Field(default="", description="What a coder agent would need to build it.")
    code_links: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Handoff(BaseModel):
    """Follow-up work for a downstream (coder) agent."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    instructions: str = ""


class ResearchResult(BaseModel):
    """Output file: written by the research CLI."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    request_id: str
    status: Literal["ok", "no_results", "error"]
    summary: str = Field(default="", description="Short plain-text summary; becomes the ticket comment.")
    papers: list[PaperSynthesis] = Field(default_factory=list)
    handoff: Handoff | None = Field(default=None, description="Set to request a follow-up ticket.")
    error: str = Field(default="", description="Human-readable reason when status is 'error'.")
