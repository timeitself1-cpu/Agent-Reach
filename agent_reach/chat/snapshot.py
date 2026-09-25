"""Where the chat's trends come from: the newest local run or the newest cloud-runner report.

* local: the newest finished run in the SQLite history (``Settings.db_path``)
* cloud: the newest successful ``cloud-runner`` run on ``chat_cloud_branch``, whose report
  artifact is downloaded once with the GitHub CLI and cached under ``chat_cache_dir``

Whichever started later is shown. Both become a :class:`PipelineReport`, the same contract
the pipeline writes, so the agent never cares where a report came from.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent_reach.config import Settings
from agent_reach.models import CategoryEnum, MacroCluster, PipelineReport

log = logging.getLogger(__name__)

GH_FALLBACK_PATHS = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "GitHub CLI" / "gh.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe",
)


@dataclass
class Snapshot:
    report: PipelineReport
    origin: str  # "local" | "cloud"
    origin_label: str  # e.g. "cloud run 36130220856 (main)"
    origin_url: str | None = None
    notes: list[str] = field(default_factory=list)  # why the other origin was not used, etc.
    history: list[dict] = field(default_factory=list)  # earlier local runs: {"started_at", "headlines"}

    @property
    def age_hours(self) -> float:
        return max(0.0, (datetime.now(timezone.utc) - self.report.started_at).total_seconds() / 3600)


# ------------------------------------------------------------------ local
def _cluster_from_row(row) -> MacroCluster | None:
    try:
        return MacroCluster(
            cluster_id=row["cluster_id"],
            headline=row["headline"],
            category=CategoryEnum(row["category"]),
            relevance_score=row["relevance_score"],
            velocity_score=row["velocity_score"],
            combined_score=row["combined_score"],
            momentum=row["momentum"] or "STEADY",
            velocity_basis=row["velocity_basis"] or "cold_start",
            summary=row["summary"] or "",
            primary_entities=json.loads(row["entities"] or "[]"),
            source_urls=json.loads(row["source_urls"] or "[]"),
            sources=json.loads(row["sources"] or "[]"),
            raw_item_count=row["raw_item_count"],
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        log.debug("skipping unreadable cluster row: %s", exc)
        return None


def load_local(settings: Settings, history_runs: int = 5) -> tuple[PipelineReport | None, list[dict], str | None]:
    """Newest finished local run as a report, plus earlier runs' headlines. Never raises."""
    path = Path(settings.db_path)
    if not path.exists():
        return None, [], f"no local history at {path}"
    from agent_reach.storage.db import TrendDatabase  # local import: sqlite setup only when needed

    db = TrendDatabase(path)
    try:
        runs = db.recent_runs(history_runs + 1)
        if not runs:
            return None, [], "local history has no finished runs"
        latest = runs[0]
        clusters = [c for c in (_cluster_from_row(r) for r in db.clusters_for_run(latest["run_id"])) if c]
        report = PipelineReport(
            run_id=latest["run_id"],
            execution_time=latest["exec_seconds"] or 0.0,
            ingested_count=latest["ingested"] or 0,
            filtered_count=latest["filtered"] or 0,
            cluster_count=len(clusters),
            macro_clusters=clusters,
            started_at=datetime.fromtimestamp(latest["started_at"], tz=timezone.utc),
            llm_mode=latest["llm_mode"] or "unknown",
        )
        history = []
        for r in runs[1:]:
            heads = [row["headline"] for row in db.clusters_for_run(r["run_id"])][:8]
            history.append({"started_at": datetime.fromtimestamp(r["started_at"], tz=timezone.utc), "headlines": heads})
        return report, history, None
    except Exception as exc:  # noqa: BLE001 - a broken DB must not stop the chat page
        return None, [], f"local history unreadable ({type(exc).__name__}: {exc})"
    finally:
        db.close()


# ------------------------------------------------------------------ cloud
def find_gh() -> str | None:
    found = shutil.which("gh")
    if found:
        return found
    for p in GH_FALLBACK_PATHS:
        if p.is_file():
            return str(p)
    return None


def _gh(gh: str, args: list[str], cwd: Path, timeout: float = 90.0) -> str:
    done = subprocess.run(
        [gh, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip()[:300] or f"gh exited {done.returncode}")
    return done.stdout


def load_cloud(settings: Settings, project_root: Path) -> tuple[PipelineReport | None, dict | None, str | None]:
    """Newest successful cloud-runner report (downloaded once, then cached). Never raises."""
    if not settings.chat_cloud_enabled:
        return None, None, "cloud reports disabled"
    gh = find_gh()
    if not gh:
        return None, None, "GitHub CLI (gh) not found, so cloud reports are unavailable"
    repo = ["--repo", settings.chat_cloud_repo] if settings.chat_cloud_repo else []
    try:
        runs = json.loads(_gh(gh, [
            "run", "list", *repo, "--workflow", settings.chat_cloud_workflow, "--branch", settings.chat_cloud_branch,
            "--status", "success", "--limit", "1", "--json", "databaseId,createdAt,url,headBranch",
        ], project_root) or "[]")
        if not runs:
            return None, None, f"no successful {settings.chat_cloud_workflow} run on {settings.chat_cloud_branch}"
        run = runs[0]
        cache = Path(settings.chat_cache_dir)
        if not cache.is_absolute():
            cache = project_root / cache
        target = cache / str(run["databaseId"])
        if not any(target.rglob("report-*.json")):
            target.mkdir(parents=True, exist_ok=True)
            _gh(gh, ["run", "download", str(run["databaseId"]), *repo, "--dir", str(target),
                     "--pattern", "agent-reach-report-*"], project_root, timeout=180.0)
        files = sorted(target.rglob("report-*.json"))
        if not files:
            return None, None, f"cloud run {run['databaseId']} has no JSON report"
        report = PipelineReport.model_validate_json(files[-1].read_text(encoding="utf-8"))
        return report, run, None
    except (RuntimeError, OSError, subprocess.TimeoutExpired, ValidationError, json.JSONDecodeError) as exc:
        return None, None, f"cloud report unavailable ({type(exc).__name__}: {str(exc)[:200]})"


# ------------------------------------------------------------------ choose
def load_latest(settings: Settings, project_root: Path, *, use_cloud: bool = True) -> tuple[Snapshot | None, list[str]]:
    """The newer of the local and cloud reports (``None`` when neither exists), plus notes on why."""
    notes: list[str] = []
    local, history, why_local = load_local(settings)
    if why_local:
        notes.append(why_local)
    cloud, run, why_cloud = (None, None, "cloud reports skipped (--no-cloud)")
    if use_cloud:
        cloud, run, why_cloud = load_cloud(settings, project_root)
    if why_cloud:
        notes.append(why_cloud)

    if cloud is not None and (local is None or cloud.started_at >= local.started_at):
        if local is not None:
            notes.append(f"local run {local.run_id} is older, so the cloud report is shown")
        label = f"cloud run {run['databaseId']} ({run.get('headBranch') or settings.chat_cloud_branch})"
        return Snapshot(cloud, "cloud", label, run.get("url"), notes, history), notes
    if local is not None:
        if cloud is not None:
            notes.append("the newest cloud report is older, so the local run is shown")
        return Snapshot(local, "local", f"local run {local.run_id}", None, notes, history), notes
    return None, notes
