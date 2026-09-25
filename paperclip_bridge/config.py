"""Bridge settings.

Paperclip injects ``PAPERCLIP_*`` variables into every heartbeat run. Bridge
behaviour is set with ``RESEARCH_BRIDGE_*`` variables, normally in the
``env`` block of the Paperclip process-adapter config.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BridgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_BRIDGE_", extra="ignore")

    # Research CLI; the bridge appends ``--in <file> --out <file>``.
    cmd: str = "python -m research_agent"
    cwd: Path | None = None
    workdir: Path = Field(default_factory=lambda: Path.home() / ".paperclip_bridge" / "runs")
    timeout_s: float = 3600.0

    # One lock file shared by every local-model worker on this machine.
    gpu_lock_path: Path = Field(default_factory=lambda: Path.home() / ".paperclip_bridge" / "gpu.lock")
    gpu_wait_s: float = 30.0

    # Agent that receives ``ResearchResult.handoff`` tickets; empty disables hand-off.
    handoff_agent_id: str = ""
    document_key: str = "research-result"
    api_timeout_s: float = 30.0

    @field_validator("cmd")
    @classmethod
    def _cmd_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("RESEARCH_BRIDGE_CMD must not be empty")
        return v

    def cmd_argv(self) -> list[str]:
        return shlex.split(self.cmd, posix=os.name != "nt")


@dataclass(frozen=True)
class PaperclipEnv:
    api_url: str
    api_key: str
    run_id: str
    agent_id: str
    company_id: str
    task_id: str = ""

    @classmethod
    def from_environ(cls, environ: dict[str, str] | None = None) -> PaperclipEnv:
        env = os.environ if environ is None else environ
        required = ("PAPERCLIP_API_URL", "PAPERCLIP_API_KEY", "PAPERCLIP_RUN_ID", "PAPERCLIP_AGENT_ID", "PAPERCLIP_COMPANY_ID")
        missing = [k for k in required if not env.get(k)]
        if missing:
            raise RuntimeError(f"missing Paperclip environment: {', '.join(missing)} (run this from a Paperclip heartbeat)")
        return cls(
            api_url=env["PAPERCLIP_API_URL"].rstrip("/"),
            api_key=env["PAPERCLIP_API_KEY"],
            run_id=env["PAPERCLIP_RUN_ID"],
            agent_id=env["PAPERCLIP_AGENT_ID"],
            company_id=env["PAPERCLIP_COMPANY_ID"],
            task_id=env.get("PAPERCLIP_TASK_ID", ""),
        )
