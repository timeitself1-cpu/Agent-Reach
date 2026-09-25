"""Offline tests for the Windows node deliverables: CLI wrapper template, setup_agent, lock self-test."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from paperclip_bridge import bridge, gpu_lock, setup_agent
from paperclip_bridge.config import split_command
from paperclip_bridge.contract import ResearchResult
from tests.test_paperclip_bridge import FakePaperclip, _issue, _run

TEMPLATES = Path(__file__).resolve().parent.parent / "paperclip_bridge" / "templates"
CLI_PATH = TEMPLATES / "research_agent" / "cli.py"

GRAPH_MODULE = '''
import os

class Paper:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class Compiled:
    def __init__(self, mode):
        self.mode = mode
    def invoke(self, state, config=None):
        assert config == {"recursion_limit": 50}
        if self.mode == "boom":
            raise RuntimeError("Ollama unreachable")
        if self.mode == "empty":
            return {"papers": [], "summary": ""}
        env_leak = sorted(k for k in os.environ if k.startswith("PAPERCLIP_"))
        papers = [
            {"title": "Toolformer 3", "link": "https://arxiv.org/abs/2", "synthesis": "Tools.",
             "implementation_notes": "Add a tool router.", "confidence": "0.9", "code_links": "https://github.com/x/y"},
            Paper(title="Weak paper", summary="Meh.", confidence=0.2),
            {"title": "", "summary": "dropped: no title"},
        ]
        out = {"papers": papers, "summary": f"topic={state['topic']} leak={env_leak}"}
        if self.mode == "explicit":
            out["handoff"] = {"title": "Build the router", "description": "From the graph."}
        return out

graph = Compiled("auto")
explicit = Compiled("explicit")
boom = Compiled("boom")
empty = Compiled("empty")

class Uncompiled:
    def compile(self):
        return Compiled("auto")

uncompiled = Uncompiled()

def factory():
    return Uncompiled()
'''


@pytest.fixture
def cli(tmp_path, monkeypatch):
    (tmp_path / "fake_graph.py").write_text(GRAPH_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_graph", None)
    spec = importlib.util.spec_from_file_location("research_cli_template", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(cli, tmp_path, monkeypatch, graph: str, **env) -> dict:
    monkeypatch.setenv("RESEARCH_AGENT_GRAPH", f"fake_graph:{graph}")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    req, out = tmp_path / "request.json", tmp_path / "result.json"
    req.write_text(json.dumps({"schema_version": "1.0", "request_id": "r-1", "title": "agents",
                               "instructions": "x"}), encoding="utf-8")
    assert cli.main(["--in", str(req), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    ResearchResult.model_validate(data)  # always satisfies the bridge contract
    return data


@pytest.mark.parametrize("graph", ["graph", "uncompiled", "factory"])
def test_cli_resolves_graph_forms_and_normalises_papers(cli, tmp_path, monkeypatch, graph):
    data = _call(cli, tmp_path, monkeypatch, graph)
    assert data["status"] == "ok" and data["request_id"] == "r-1"
    assert [p["title"] for p in data["papers"]] == ["Toolformer 3", "Weak paper"]
    first = data["papers"][0]
    assert first["url"] == "https://arxiv.org/abs/2" and first["summary"] == "Tools."
    assert first["confidence"] == 0.9 and first["code_links"] == ["https://github.com/x/y"]
    assert data["summary"].startswith("topic=agents")


def test_cli_auto_handoff_uses_best_buildable_paper(cli, tmp_path, monkeypatch):
    data = _call(cli, tmp_path, monkeypatch, "graph")
    assert data["handoff"]["title"] == "Prototype: Toolformer 3"
    assert "Add a tool router." in data["handoff"]["instructions"]
    assert _call(cli, tmp_path, monkeypatch, "graph", RESEARCH_AGENT_AUTO_HANDOFF="0")["handoff"] is None
    assert _call(cli, tmp_path, monkeypatch, "graph", RESEARCH_AGENT_HANDOFF_MIN_CONFIDENCE="0.95")["handoff"] is None


def test_cli_explicit_handoff_wins(cli, tmp_path, monkeypatch):
    data = _call(cli, tmp_path, monkeypatch, "explicit")
    assert data["handoff"] == {"title": "Build the router", "instructions": "From the graph."}


@pytest.mark.parametrize("graph, expected", [("boom", "RuntimeError: Ollama unreachable"),
                                             ("missing_attr", "AttributeError")])
def test_cli_maps_failures_to_error_status(cli, tmp_path, monkeypatch, graph, expected):
    data = _call(cli, tmp_path, monkeypatch, graph)
    assert data["status"] == "error" and expected in data["error"] and data["papers"] == []


def test_cli_empty_graph_output_is_no_results(cli, tmp_path, monkeypatch):
    data = _call(cli, tmp_path, monkeypatch, "empty")
    assert data["status"] == "no_results" and data["handoff"] is None


def test_cli_bad_request_exits_2(cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli.main(["--in", str(bad), "--out", str(tmp_path / "o.json")]) == 2
    bad.write_text(json.dumps({"title": "no id"}), encoding="utf-8")
    assert cli.main(["--in", str(bad), "--out", str(tmp_path / "o.json")]) == 2


def test_bridge_runs_template_cli_end_to_end_without_paperclip_env(tmp_path, monkeypatch):
    (tmp_path / "fake_graph.py").write_text(GRAPH_MODULE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("RESEARCH_AGENT_GRAPH", "fake_graph:graph")
    monkeypatch.setenv("PAPERCLIP_API_KEY", "secret-run-token")
    fake = FakePaperclip([_issue()])
    code, summary = _run(tmp_path, fake, f'"{sys.executable}" "{CLI_PATH}"')
    assert code == 0 and summary["summary"].startswith("done: 2 paper(s)")
    assert fake.issues["iss-1"]["status"] == "done"
    [child] = fake.created
    assert child["title"] == "Prototype: Toolformer 3"
    doc = fake.documents[("iss-1", "research-result")]["body"]
    assert "leak=[]" in doc  # the CLI never sees PAPERCLIP_* variables


# ------------------------------------------------------------ command splitting
@pytest.mark.parametrize("cmd, expected", [
    (r'"C:\Program Files\Python312\python.exe" -m research_agent.cli',
     [r"C:\Program Files\Python312\python.exe", "-m", "research_agent.cli"]),
    (r"C:\venv\Scripts\python.exe -m research_agent.cli", [r"C:\venv\Scripts\python.exe", "-m", "research_agent.cli"]),
])
def test_split_command_windows(cmd, expected):
    assert split_command(cmd, windows=True) == expected


def test_split_command_posix():
    assert split_command('"/opt/my py/python" -m x', windows=False) == ["/opt/my py/python", "-m", "x"]


# ------------------------------------------------------------ paperclip.yaml + setup_agent
def _config(**agent_env) -> dict:
    cfg = yaml.safe_load((TEMPLATES / "paperclip.yaml").read_text(encoding="utf-8"))
    cfg["agent"]["adapterConfig"]["env"].update(agent_env)
    return cfg


def test_template_yaml_has_required_settings():
    cfg = _config()
    hb = cfg["agent"]["runtimeConfig"]["heartbeat"]
    assert hb == {"enabled": True, "intervalSec": 900, "maxConcurrentRuns": 1, "wakeOnDemand": True}
    env = cfg["agent"]["adapterConfig"]["env"]
    assert split_command(env["RESEARCH_BRIDGE_CMD"], windows=True)[1:] == ["-m", "research_agent.cli"]
    assert cfg["agent"]["adapterConfig"]["args"] == ["-m", "paperclip_bridge"]


def test_template_placeholder_is_refused_until_filled(tmp_path):
    path = tmp_path / "paperclip.yaml"
    path.write_text((TEMPLATES / "paperclip.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(setup_agent.ConfigError, match="placeholder"):
        setup_agent.load_config(path)
    path.write_text(path.read_text(encoding="utf-8").replace("<coder-agent-uuid>", ""), encoding="utf-8")
    assert setup_agent.load_config(path)["agent"]["name"] == "Research Agent"


@pytest.mark.parametrize("mutate, message", [
    (lambda a: a["runtimeConfig"]["heartbeat"].update(maxConcurrentRuns=20), "maxConcurrentRuns"),
    (lambda a: a["adapterConfig"]["env"].pop("RESEARCH_BRIDGE_CMD"), "RESEARCH_BRIDGE_CMD"),
    (lambda a: a["adapterConfig"]["env"].update(RESEARCH_BRIDGE_TIMEOUT_S=3600), "must be a string"),
    (lambda a: a.update(adapterType="http"), "process"),
])
def test_validate_agent_rejects_unsafe_config(mutate, message):
    agent = _config(RESEARCH_BRIDGE_HANDOFF_AGENT_ID="")["agent"]
    mutate(agent)
    with pytest.raises(setup_agent.ConfigError, match=message):
        setup_agent.validate_agent(agent)


class FakeBoard:
    def __init__(self, agents: list[dict], statuses: list[str] | None = None):
        self.agents = agents
        self.statuses = list(statuses or [])
        self.requests: list[tuple[str, str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((request.method, path, body))
        if path == "/api/companies":
            return httpx.Response(200, json=[{"id": "co-1", "name": "Lab"}])
        if path == "/api/companies/co-1/agents" and request.method == "GET":
            return httpx.Response(200, json=self.agents)
        if path == "/api/companies/co-1/agents" and request.method == "POST":
            return httpx.Response(201, json={"id": "new-agent", **body})
        if path.startswith("/api/agents/") and request.method == "PATCH":
            return httpx.Response(200, json=body)
        if path == "/api/companies/co-1/issues":
            return httpx.Response(201, json={"id": "iss-9", "identifier": "LAB-9", "status": "todo"})
        if path == "/api/issues/iss-9":
            return httpx.Response(200, json={"id": "iss-9", "status": self.statuses.pop(0)})
        if path == "/api/issues/iss-9/comments":
            return httpx.Response(200, json=[{"body": "Research finished."}])
        return httpx.Response(404, json={"error": path})


def _board(fake: FakeBoard) -> setup_agent.BoardClient:
    return setup_agent.BoardClient("http://pc", transport=httpx.MockTransport(fake.handler))


def test_apply_creates_then_updates_agent():
    cfg = _config(RESEARCH_BRIDGE_HANDOFF_AGENT_ID="")
    fake = FakeBoard([])
    assert setup_agent.apply(cfg, _board(fake), out=io.StringIO()) == "new-agent"
    method, path, body = fake.requests[-1]
    assert (method, path) == ("POST", "/api/companies/co-1/agents")
    assert body["role"] == "researcher" and body["runtimeConfig"]["heartbeat"]["maxConcurrentRuns"] == 1

    fake = FakeBoard([{"id": "a-1", "name": "Research Agent"}])
    assert setup_agent.apply(cfg, _board(fake), out=io.StringIO()) == "a-1"
    method, path, body = fake.requests[-1]
    assert (method, path) == ("PATCH", "/api/agents/a-1") and body["replaceAdapterConfig"] is True
    assert body["adapterConfig"]["env"]["RESEARCH_BRIDGE_CMD"].endswith("-m research_agent.cli")


@pytest.mark.parametrize("statuses, code, verdict", [
    (["in_progress", "done"], 0, "PASS"),
    (["in_progress", "blocked"], 1, "FAIL"),
])
def test_smoke_assigns_ticket_and_waits(statuses, code, verdict):
    cfg = _config(RESEARCH_BRIDGE_HANDOFF_AGENT_ID="")
    fake = FakeBoard([{"id": "a-1", "name": "Research Agent"}], statuses)
    out = io.StringIO()
    assert setup_agent.smoke(cfg, _board(fake), timeout_s=60, poll_s=0, out=out) == code
    create = next(b for m, p, b in fake.requests if p == "/api/companies/co-1/issues")
    assert create["assigneeAgentId"] == "a-1" and create["status"] == "todo"
    assert verdict in out.getvalue() and "Research finished." in out.getvalue()


def test_smoke_times_out():
    fake = FakeBoard([{"id": "a-1", "name": "Research Agent"}], ["in_progress"] * 5)
    out = io.StringIO()
    code = setup_agent.smoke(_config(RESEARCH_BRIDGE_HANDOFF_AGENT_ID=""), _board(fake), timeout_s=0, poll_s=0, out=out)
    assert code == 1 and "timed out" in out.getvalue()


# ------------------------------------------------------------ lock self-test + CLI subcommands
def test_lock_selftest_passes_across_processes(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    assert gpu_lock.selftest(tmp_path / "gpu.lock", hold_s=1.0) == []


def test_lock_selftest_subcommand_reports_backend(tmp_path):
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run([sys.executable, "-m", "paperclip_bridge", "lock-selftest", str(tmp_path / "g.lock")],
                          cwd=root, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(root)})
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 0 and report["ok"] and report["backend"] == gpu_lock.BACKEND


def test_validate_subcommand(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"request_id": "r", "status": "ok"}), encoding="utf-8")
    assert bridge.main(["validate", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"status": "maybe"}), encoding="utf-8")
    assert bridge.main(["validate", str(bad)]) == 1
