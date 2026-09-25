"""Apply ``paperclip.yaml`` to a local Paperclip instance and run a live smoke test.

    python -m paperclip_bridge.setup_agent apply --config paperclip.yaml
    python -m paperclip_bridge.setup_agent smoke --config paperclip.yaml [--timeout 1800]

``apply`` creates the agent (matched by name) or replaces its adapter and
runtime config. ``smoke`` assigns a small research ticket to the agent, which
wakes it, then waits for the ticket to reach ``done`` or ``blocked``.

A local Paperclip (``local_trusted`` mode) needs no credentials. For an
authenticated instance, set ``PAPERCLIP_BOARD_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

TERMINAL_STATUSES = {"done", "blocked", "cancelled"}


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise ConfigError("PyYAML is required: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not isinstance(data.get("agent"), dict):
        raise ConfigError(f"{path}: expected a top-level 'agent' mapping")
    validate_agent(data["agent"])
    return data


def validate_agent(agent: dict[str, Any]) -> None:
    if not agent.get("name"):
        raise ConfigError("agent.name is required")
    if agent.get("adapterType") != "process":
        raise ConfigError("agent.adapterType must be 'process'")
    adapter = agent.get("adapterConfig") or {}
    if not adapter.get("command"):
        raise ConfigError("agent.adapterConfig.command is required")
    env = adapter.get("env") or {}
    for key, value in env.items():
        if not isinstance(value, str):
            raise ConfigError(f"agent.adapterConfig.env.{key} must be a string (quote it in YAML)")
        if "<" in value and ">" in value:
            raise ConfigError(f"agent.adapterConfig.env.{key} still holds a placeholder: {value!r}")
    if not env.get("RESEARCH_BRIDGE_CMD"):
        raise ConfigError("agent.adapterConfig.env.RESEARCH_BRIDGE_CMD is required")
    heartbeat = ((agent.get("runtimeConfig") or {}).get("heartbeat") or {})
    if heartbeat.get("maxConcurrentRuns") != 1:
        raise ConfigError("agent.runtimeConfig.heartbeat.maxConcurrentRuns must be 1 (one GPU job per agent)")


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "agents", "companies", "comments"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


class BoardClient:
    def __init__(self, api_url: str, *, token: str = "", transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=api_url.rstrip("/"), headers=headers, timeout=30.0, transport=transport)

    def request(self, method: str, path: str, **kw: Any) -> Any:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else None

    def company_id(self, configured: str) -> str:
        if configured:
            return configured
        companies = _items(self.request("GET", "/api/companies"))
        if len(companies) != 1:
            names = ", ".join(f"{c.get('name')} ({c.get('id')})" for c in companies) or "none"
            raise ConfigError(f"set paperclip.companyId; found {len(companies)} companies: {names}")
        return companies[0]["id"]

    def find_agent(self, company_id: str, name: str) -> dict[str, Any] | None:
        agents = _items(self.request("GET", f"/api/companies/{company_id}/agents"))
        return next((a for a in agents if a.get("name") == name), None)


def apply(cfg: dict[str, Any], client: BoardClient, out=sys.stdout) -> str:
    agent = cfg["agent"]
    company_id = client.company_id((cfg.get("paperclip") or {}).get("companyId", ""))
    command = Path(agent["adapterConfig"]["command"])
    if command.is_absolute() and not command.exists():
        out.write(f"warning: adapterConfig.command does not exist on this machine: {command}\n")
    fields = {
        "adapterType": agent["adapterType"],
        "adapterConfig": agent["adapterConfig"],
        "runtimeConfig": agent.get("runtimeConfig") or {},
    }
    existing = client.find_agent(company_id, agent["name"])
    if existing:
        client.request("PATCH", f"/api/agents/{existing['id']}", json={**fields, "replaceAdapterConfig": True})
        agent_id = existing["id"]
        out.write(f"updated agent {agent['name']} ({agent_id})\n")
    else:
        body = {"name": agent["name"], **fields}
        for key in ("role", "title", "capabilities"):
            if agent.get(key):
                body[key] = agent[key]
        agent_id = client.request("POST", f"/api/companies/{company_id}/agents", json=body)["id"]
        out.write(f"created agent {agent['name']} ({agent_id})\n")
    return agent_id


def smoke(cfg: dict[str, Any], client: BoardClient, *, timeout_s: float, poll_s: float = 10.0, out=sys.stdout) -> int:
    agent = cfg["agent"]
    company_id = client.company_id((cfg.get("paperclip") or {}).get("companyId", ""))
    existing = client.find_agent(company_id, agent["name"])
    if not existing:
        raise ConfigError(f"agent {agent['name']!r} not found; run 'apply' first")
    smoke_cfg = cfg.get("smokeTest") or {}
    issue = client.request("POST", f"/api/companies/{company_id}/issues", json={
        "title": smoke_cfg.get("title") or "Paperclip bridge smoke test",
        "description": smoke_cfg.get("description") or "Find one recent AI paper on agent tool use and summarise it.",
        "status": "todo",
        "priority": "low",
        "assigneeAgentId": existing["id"],
    })
    label = issue.get("identifier") or issue["id"]
    out.write(f"created {label}; assignment should wake {agent['name']}\n")
    deadline = time.monotonic() + timeout_s
    status = issue.get("status")
    while status not in TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(poll_s)
        status = client.request("GET", f"/api/issues/{issue['id']}").get("status")
        out.write(f"  {label}: {status}\n")
        out.flush()
    comments = _items(client.request("GET", f"/api/issues/{issue['id']}/comments"))
    if comments:
        out.write("latest comment:\n" + str(comments[-1].get("body", "")).strip() + "\n")
    if status == "done":
        out.write(f"PASS: {label} is done\n")
        return 0
    out.write(f"FAIL: {label} ended as {status!r}" + (" (timed out)" if status not in TERMINAL_STATUSES else "") + "\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m paperclip_bridge.setup_agent")
    ap.add_argument("command", choices=["apply", "smoke"])
    ap.add_argument("--config", type=Path, default=Path("paperclip.yaml"))
    ap.add_argument("--api-url", default="")
    ap.add_argument("--timeout", type=float, default=1800.0, help="smoke: seconds to wait for the ticket")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        api_url = args.api_url or (cfg.get("paperclip") or {}).get("apiUrl") or "http://localhost:3100"
        client = BoardClient(api_url, token=os.environ.get("PAPERCLIP_BOARD_TOKEN", ""))
        if args.command == "apply":
            apply(cfg, client)
            return 0
        return smoke(cfg, client, timeout_s=args.timeout)
    except (ConfigError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"setup_agent: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
