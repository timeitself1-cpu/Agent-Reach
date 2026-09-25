"""Minimal Paperclip REST client for one heartbeat run."""

from __future__ import annotations

from typing import Any

import httpx

from .config import PaperclipEnv

CLAIMABLE_STATUSES = ["todo", "in_progress"]
_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class PaperclipClient:
    def __init__(self, env: PaperclipEnv, *, timeout_s: float = 30.0, transport: httpx.BaseTransport | None = None):
        self.env = env
        self._http = httpx.Client(
            base_url=env.api_url,
            headers={
                "Authorization": f"Bearer {env.api_key}",
                "X-Paperclip-Run-Id": env.run_id,
                "Accept": "application/json",
            },
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> PaperclipClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def candidate_issue_ids(self) -> list[str]:
        """Assigned, unblocked tickets no other run is executing, best first."""
        r = self._http.get("/api/agents/me/inbox-lite")
        r.raise_for_status()
        rows = [
            row for row in r.json()
            if row.get("status") in CLAIMABLE_STATUSES
            and row.get("dependencyReady", True)
            and (row.get("activeRun") or {}).get("id", self.env.run_id) == self.env.run_id
        ]
        # in_progress first (a previous run of ours died mid-way), then priority, then oldest.
        rows.sort(key=lambda row: (
            row.get("status") != "in_progress",
            _PRIORITY_RANK.get(row.get("priority", "medium"), 2),
            row.get("updatedAt") or "",
        ))
        return [row["id"] for row in rows]

    def checkout(self, issue_id: str) -> bool:
        """Atomically claim the ticket. False when another agent or run holds it."""
        r = self._http.post(
            f"/api/issues/{issue_id}/checkout",
            json={"agentId": self.env.agent_id, "expectedStatuses": CLAIMABLE_STATUSES},
        )
        if r.status_code == 409:
            return False
        r.raise_for_status()
        return True

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        r = self._http.get(f"/api/issues/{issue_id}")
        r.raise_for_status()
        return r.json()

    def put_document(self, issue_id: str, key: str, title: str, markdown: str) -> None:
        r = self._http.put(
            f"/api/issues/{issue_id}/documents/{key}",
            json={"title": title, "format": "markdown", "body": markdown},
        )
        r.raise_for_status()

    def update_issue(self, issue_id: str, **fields: Any) -> dict[str, Any]:
        r = self._http.patch(f"/api/issues/{issue_id}", json=fields)
        r.raise_for_status()
        return r.json()

    def create_issue(self, **fields: Any) -> dict[str, Any]:
        r = self._http.post(f"/api/companies/{self.env.company_id}/issues", json=fields)
        r.raise_for_status()
        return r.json()
