"""Offline tests for paperclip_bridge: fake Paperclip API (httpx.MockTransport) + fake research CLI."""

from __future__ import annotations

import io
import json
import sys
import textwrap

import httpx
import pytest

from paperclip_bridge import bridge
from paperclip_bridge.client import PaperclipClient
from paperclip_bridge.config import BridgeSettings, PaperclipEnv
from paperclip_bridge.contract import ResearchResult
from paperclip_bridge.gpu_lock import GpuLock

AGENT, CODER, COMPANY, RUN = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "run-1",
)


class FakePaperclip:
    """In-memory Paperclip with the endpoints the bridge uses."""

    def __init__(self, issues: list[dict]):
        self.issues = {i["id"]: {"priority": "medium", "updatedAt": "2026-01-01", **i} for i in issues}
        self.documents: dict[tuple[str, str], dict] = {}
        self.comments: list[tuple[str, str]] = []
        self.created: list[dict] = []
        self.locked: set[str] = set()
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer key"
        assert request.headers["X-Paperclip-Run-Id"] == RUN
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else {}
        self.calls.append(f"{method} {path}")
        if path == "/api/agents/me/inbox-lite":
            return httpx.Response(200, json=[
                {k: i.get(k) for k in ("id", "identifier", "title", "status", "priority", "updatedAt", "activeRun")}
                | {"dependencyReady": True}
                for i in self.issues.values()
                if i["status"] in ("todo", "in_progress", "blocked") and i.get("assigneeAgentId", AGENT) == AGENT
            ])
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "issues"]:
            issue = self.issues[parts[2]]
            if parts[3:] == ["checkout"]:
                assert body == {"agentId": AGENT, "expectedStatuses": ["todo", "in_progress"]}
                if issue["id"] in self.locked or issue["status"] not in body["expectedStatuses"]:
                    return httpx.Response(409, json={"error": "conflict"})
                issue["status"] = "in_progress"
                return httpx.Response(200, json=issue)
            if parts[3:4] == ["documents"] and method == "PUT":
                assert body["format"] == "markdown"
                self.documents[(issue["id"], parts[4])] = body
                return httpx.Response(200, json={"key": parts[4]})
            if method == "PATCH":
                if "comment" in body:
                    self.comments.append((issue["id"], body.pop("comment")))
                issue.update(body)
                return httpx.Response(200, json=issue)
            return httpx.Response(200, json=issue)
        if path == f"/api/companies/{COMPANY}/issues" and method == "POST":
            child = {"id": f"child-{len(self.created) + 1}", "identifier": f"RES-{100 + len(self.created)}", **body}
            self.created.append(child)
            self.issues[child["id"]] = child
            return httpx.Response(201, json=child)
        return httpx.Response(404, json={"error": path})


def _cli(tmp_path, body: str) -> str:
    script = tmp_path / "fake_cli.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        args = sys.argv[1:]
        req = json.load(open(args[args.index("--in") + 1], encoding="utf-8"))
        out = args[args.index("--out") + 1]
    """) + textwrap.dedent(body), encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


OK_CLI = """
    json.dump({"request_id": req["request_id"], "status": "ok", "summary": "Two agent papers.",
               "papers": [{"title": "ReAct 2", "url": "https://arxiv.org/abs/1", "summary": "S",
                           "implementation_notes": "Build a tool loop."}],
               "handoff": {"title": "Prototype ReAct 2", "instructions": "Implement it."}},
              open(out, "w", encoding="utf-8"))
"""


def _run(tmp_path, fake: FakePaperclip, cmd: str, **settings_kw) -> tuple[int, dict]:
    env = PaperclipEnv(api_url="http://pc", api_key="key", run_id=RUN, agent_id=AGENT, company_id=COMPANY)
    kw = dict(cmd=cmd, workdir=tmp_path / "runs", gpu_lock_path=tmp_path / "gpu.lock",
              gpu_wait_s=0.0, handoff_agent_id=CODER, timeout_s=60)
    settings = BridgeSettings(**{**kw, **settings_kw})
    out = io.StringIO()
    with PaperclipClient(env, transport=httpx.MockTransport(fake.handler)) as client:
        code = bridge.run_heartbeat(settings, env, client, out=out)
    return code, json.loads(out.getvalue())


def _issue(i="iss-1", status="todo", **kw):
    return {"id": i, "identifier": f"RES-{i[-1]}", "title": "Weekly agent papers", "description": "Focus on tool use.",
            "status": status, "goalId": "goal-1", **kw}


def test_happy_path_writes_document_hands_off_and_closes(tmp_path):
    fake = FakePaperclip([_issue()])
    code, summary = _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    assert code == 0 and summary["issueId"] == "iss-1" and summary["summary"].startswith("done")
    assert fake.issues["iss-1"]["status"] == "done"
    doc = fake.documents[("iss-1", "research-result")]
    assert "ReAct 2" in doc["body"] and '"request_id"' in doc["body"]
    [child] = fake.created
    assert child["parentId"] == "iss-1" and child["assigneeAgentId"] == CODER
    assert child["status"] == "todo" and child["goalId"] == "goal-1"
    assert "Build a tool loop." in child["description"]
    assert "RES-100" in fake.comments[-1][1]
    request = json.loads(next((tmp_path / "runs").rglob("request.json")).read_text())
    assert request["title"] == "Weekly agent papers" and request["instructions"] == "Focus on tool use."
    assert not any(k.startswith("paperclip") or "issue" in k.lower() for k in request)


def test_handoff_is_created_once_across_retries(tmp_path):
    fake = FakePaperclip([_issue()])
    _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    fake.issues["iss-1"]["status"] = "todo"  # e.g. a previous run died before PATCH done
    _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    assert len(fake.created) == 1


def test_checkout_conflict_falls_through_to_next_candidate(tmp_path):
    fake = FakePaperclip([_issue("iss-1", priority="high"), _issue("iss-2")])
    fake.locked.add("iss-1")
    code, summary = _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    assert code == 0 and summary["issueId"] == "iss-2"
    assert fake.issues["iss-1"]["status"] == "todo"


def test_skips_tickets_another_run_is_executing(tmp_path):
    fake = FakePaperclip([_issue("iss-1", status="in_progress", activeRun={"id": "other-run"})])
    code, summary = _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    assert code == 0 and "No claimable" in summary["summary"]
    assert not any("checkout" in c for c in fake.calls)


@pytest.mark.parametrize("body, reason", [
    ("sys.exit(3)", "exited with code 3"),
    ("sys.stderr.write('boom trace'); sys.exit(1)", "boom trace"),
    ("pass", "wrote no result file"),
    ("open(out, 'w').write('{\"status\": \"ok\"}')", "invalid result file"),
    ("json.dump({'request_id': 'wrong', 'status': 'ok'}, open(out, 'w'))", "does not match"),
    ("json.dump({'request_id': req['request_id'], 'status': 'error', 'error': 'arXiv down'}, open(out, 'w'))",
     "arXiv down"),
])
def test_cli_failures_block_the_ticket(tmp_path, body, reason):
    fake = FakePaperclip([_issue()])
    code, summary = _run(tmp_path, fake, _cli(tmp_path, body))
    assert code == 0 and summary["summary"].startswith("blocked")
    issue = fake.issues["iss-1"]
    assert issue["status"] == "blocked" and issue["unblockDescriptor"]["owner"] == "board"
    assert reason in fake.comments[-1][1]
    assert not fake.created and not fake.documents


def test_timeout_blocks_the_ticket(tmp_path):
    fake = FakePaperclip([_issue()])
    code, summary = _run(tmp_path, fake, _cli(tmp_path, "import time; time.sleep(5)"), timeout_s=0.5)
    assert fake.issues["iss-1"]["status"] == "blocked" and "timed out" in fake.comments[-1][1]


def test_busy_gpu_leaves_tickets_untouched(tmp_path):
    fake = FakePaperclip([_issue()])
    with GpuLock(tmp_path / "gpu.lock", 0) as held:
        assert held
        code, summary = _run(tmp_path, fake, _cli(tmp_path, OK_CLI))
    assert code == 0 and "GPU busy" in summary["summary"]
    assert fake.issues["iss-1"]["status"] == "todo"
    assert not any("checkout" in c for c in fake.calls)


def test_handoff_without_agent_is_reported_not_created(tmp_path):
    fake = FakePaperclip([_issue()])
    _run(tmp_path, fake, _cli(tmp_path, OK_CLI), handoff_agent_id="")
    assert not fake.created and "not set" in fake.comments[-1][1]
    assert fake.issues["iss-1"]["status"] == "done"


def test_env_requires_paperclip_variables():
    with pytest.raises(RuntimeError, match="PAPERCLIP_RUN_ID"):
        PaperclipEnv.from_environ({"PAPERCLIP_API_URL": "x", "PAPERCLIP_API_KEY": "k",
                                   "PAPERCLIP_AGENT_ID": AGENT, "PAPERCLIP_COMPANY_ID": COMPANY})


def test_main_without_paperclip_env_fails_cleanly(monkeypatch, capsys):
    for k in ("PAPERCLIP_API_URL", "PAPERCLIP_API_KEY", "PAPERCLIP_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    assert bridge.main([]) == 1
    assert "missing Paperclip environment" in capsys.readouterr().err


def test_schema_command_and_contract_defaults(capsys):
    assert bridge.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert set(schema) == {"request", "result"}
    result = ResearchResult(request_id="r", status="no_results")
    assert result.schema_version == "1.0" and result.papers == [] and result.handoff is None
