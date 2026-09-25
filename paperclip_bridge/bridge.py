"""One Paperclip heartbeat: claim a ticket, run the research CLI, report back."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from .client import PaperclipClient
from .config import BridgeSettings, PaperclipEnv
from .contract import ResearchRequest, ResearchResult
from .gpu_lock import GpuLock

DOCUMENT_MAX_CHARS = 500_000
TAIL_CHARS = 1500


@dataclass
class CliOutcome:
    run_dir: Path
    result: ResearchResult | None = None
    failure: str = ""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "issue"


def _tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-TAIL_CHARS:].strip()
    except OSError:
        return ""


def run_cli(settings: BridgeSettings, issue: dict[str, Any], run_id: str) -> CliOutcome:
    """Write request.json, run the CLI, read result.json. Never raises for CLI problems."""
    label = issue.get("identifier") or issue["id"]
    run_dir = settings.workdir / _safe_name(label) / _safe_name(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    request = ResearchRequest(
        request_id=f"{label}-{run_id}",
        title=issue.get("title") or label,
        instructions=issue.get("description") or "",
    )
    req_path, res_path = run_dir / "request.json", run_dir / "result.json"
    req_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    res_path.unlink(missing_ok=True)
    outcome = CliOutcome(run_dir=run_dir)

    argv = settings.cmd_argv() + ["--in", str(req_path), "--out", str(res_path)]
    stdout_log, stderr_log = run_dir / "stdout.log", run_dir / "stderr.log"
    try:
        with open(stdout_log, "wb") as so, open(stderr_log, "wb") as se:
            proc = subprocess.run(
                argv, cwd=settings.cwd, stdout=so, stderr=se,
                stdin=subprocess.DEVNULL, timeout=settings.timeout_s,
            )
    except subprocess.TimeoutExpired:
        outcome.failure = f"research CLI timed out after {settings.timeout_s:.0f}s"
        return outcome
    except OSError as exc:
        outcome.failure = f"could not start research CLI {argv[0]!r}: {exc}"
        return outcome

    if proc.returncode != 0:
        outcome.failure = f"research CLI exited with code {proc.returncode}"
        tail = _tail(stderr_log)
        if tail:
            outcome.failure += f"\n\nstderr (tail):\n```\n{tail}\n```"
        return outcome
    try:
        result = ResearchResult.model_validate_json(res_path.read_bytes())
    except FileNotFoundError:
        outcome.failure = "research CLI exited 0 but wrote no result file"
        return outcome
    except ValidationError as exc:
        outcome.failure = f"research CLI wrote an invalid result file:\n```\n{exc}\n```"
        return outcome
    if result.request_id != request.request_id:
        outcome.failure = f"result.request_id {result.request_id!r} does not match {request.request_id!r}"
        return outcome
    outcome.result = result
    return outcome


def render_markdown(result: ResearchResult) -> str:
    lines = [f"# Research result ({result.status})", "", result.summary or "_No summary._", ""]
    for i, p in enumerate(result.papers, 1):
        link = f" ([link]({p.url}))" if p.url else ""
        lines += [f"## {i}. {p.title}{link}", "", p.summary, ""]
        if p.key_claims:
            lines += ["**Key claims**", ""] + [f"- {c}" for c in p.key_claims] + [""]
        if p.methods:
            lines += [f"**Methods:** {p.methods}", ""]
        if p.implementation_notes:
            lines += [f"**Implementation notes:** {p.implementation_notes}", ""]
        if p.code_links:
            lines += ["**Code:** " + ", ".join(p.code_links), ""]
        lines += [f"_Confidence: {p.confidence:.2f}_", ""]
    payload = result.model_dump_json(indent=2)
    block = ["## Machine-readable payload", "", "```json", payload, "```", ""]
    text = "\n".join(lines)
    if len(text) + len(payload) + 64 <= DOCUMENT_MAX_CHARS:
        text += "\n" + "\n".join(block)
    return text[:DOCUMENT_MAX_CHARS]


def _create_handoff(client: PaperclipClient, settings: BridgeSettings, issue: dict[str, Any], result: ResearchResult) -> str:
    """Create the follow-up ticket once per research ticket. Returns its identifier."""
    marker = settings.workdir / "handoffs" / f"{_safe_name(issue['id'])}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))["identifier"]
    handoff = result.handoff
    if handoff is None:
        raise ValueError("result has no handoff")
    parent_label = issue.get("identifier") or issue["id"]
    body = [handoff.instructions.strip(), "",
            f"Research source: {parent_label}, document `{settings.document_key}`.", ""]
    for p in result.papers:
        body.append(f"- {p.title}" + (f" <{p.url}>" if p.url else ""))
        if p.implementation_notes:
            body.append(f"  - {p.implementation_notes}")
    fields: dict[str, Any] = {
        "title": handoff.title,
        "description": "\n".join(body).strip(),
        "status": "todo",
        "parentId": issue["id"],
        "assigneeAgentId": settings.handoff_agent_id,
        "priority": issue.get("priority") or "medium",
    }
    for key in ("goalId", "projectId"):
        if issue.get(key):
            fields[key] = issue[key]
    child = client.create_issue(**fields)
    identifier = child.get("identifier") or child["id"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"id": child["id"], "identifier": identifier}), encoding="utf-8")
    return identifier


def report(client: PaperclipClient, settings: BridgeSettings, issue: dict[str, Any], outcome: CliOutcome) -> str:
    """Write the outcome to Paperclip. Returns a one-line summary for the run log."""
    issue_id = issue["id"]
    result = outcome.result
    if result is None or result.status == "error":
        reason = outcome.failure or result.error or "research CLI reported an error"
        client.update_issue(
            issue_id,
            status="blocked",
            comment=f"Research failed: {reason}\n\nRun files: `{outcome.run_dir}`",
            unblockDescriptor={"owner": "board", "action": "Fix the research CLI failure, then move this ticket back to todo."},
        )
        return f"blocked: {reason.splitlines()[0]}"

    client.put_document(issue_id, settings.document_key, "Research result", render_markdown(result))
    comment = [result.summary or f"Research finished ({result.status}).",
               "", f"Full result: document `{settings.document_key}`."]
    if result.handoff is not None:
        if settings.handoff_agent_id:
            comment.append(f"Hand-off ticket: {_create_handoff(client, settings, issue, result)}.")
        else:
            comment.append("Hand-off requested but RESEARCH_BRIDGE_HANDOFF_AGENT_ID is not set; no ticket created.")
    updated = client.update_issue(issue_id, status="done", comment="\n".join(comment))
    if updated.get("status") not in (None, "done"):
        return f"warning: asked for done, Paperclip reports {updated.get('status')!r}"
    return f"done: {len(result.papers)} paper(s), status {result.status}"


def run_heartbeat(settings: BridgeSettings, env: PaperclipEnv, client: PaperclipClient, out: TextIO = sys.stdout) -> int:
    def emit(summary: str, **extra: Any) -> int:
        out.write(json.dumps({"summary": summary, **extra}) + "\n")
        return 0

    candidates = [env.task_id] if env.task_id else []
    candidates += [i for i in client.candidate_issue_ids() if i not in candidates]
    if not candidates:
        return emit("No claimable research tickets.")

    # Take the GPU before claiming anything, so a busy GPU leaves tickets untouched.
    with GpuLock(settings.gpu_lock_path, settings.gpu_wait_s) as got_gpu:
        if not got_gpu:
            return emit("GPU busy; tickets left for the next heartbeat.")
        issue_id = next((i for i in candidates if client.checkout(i)), None)
        if issue_id is None:
            return emit("Every candidate ticket was claimed by another run.")
        issue = client.get_issue(issue_id)
        outcome = run_cli(settings, issue, env.run_id)
    return emit(report(client, settings, issue, outcome), issueId=issue_id)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["schema"]:
        print(json.dumps({"request": ResearchRequest.model_json_schema(),
                          "result": ResearchResult.model_json_schema()}, indent=2))
        return 0
    if args:
        print("usage: python -m paperclip_bridge [schema]", file=sys.stderr)
        return 2
    try:
        settings = BridgeSettings()
        env = PaperclipEnv.from_environ()
        with PaperclipClient(env, timeout_s=settings.api_timeout_s) as client:
            return run_heartbeat(settings, env, client)
    except Exception as exc:  # bridge failure: non-zero exit marks the Paperclip run failed
        print(f"paperclip_bridge: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
