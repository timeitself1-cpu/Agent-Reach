"""Local web page for Reach chat (standard library HTTP server, bound to 127.0.0.1 only).

    GET  /              the page (trends on the left, chat on the right)
    GET  /api/snapshot  the report being discussed, as JSON
    POST /api/refresh   look again for a newer local or cloud report
    POST /api/chat      {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
                        -> newline-delimited JSON events streamed as the model writes
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from agent_reach.chat.agent import ReachAgent
from agent_reach.chat.snapshot import Snapshot, load_latest
from agent_reach.config import Settings, get_settings

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_BODY_BYTES = 256_000


def snapshot_json(snap: Snapshot | None, notes: list[str], model: str) -> dict:
    if snap is None:
        return {"available": False, "notes": notes, "model": model, "trends": [], "source_stats": []}
    r = snap.report
    return {
        "available": True,
        "origin": snap.origin,
        "origin_label": snap.origin_label,
        "origin_url": snap.origin_url,
        "run_id": r.run_id,
        "started_at": r.started_at.isoformat(),
        "age_hours": round(snap.age_hours, 2),
        "llm_mode": r.llm_mode,
        "model": model,
        "notes": notes,
        "trends": [
            {
                "rank": i,
                "headline": c.headline,
                "category": c.category.value,
                "relevance": c.relevance_score,
                "velocity": round(c.velocity_score, 1),
                "momentum": c.momentum,
                "summary": c.summary,
                "entities": c.primary_entities,
                "urls": c.source_urls,
                "sources": c.sources,
                "items": c.raw_item_count,
            }
            for i, c in enumerate(r.macro_clusters, start=1)
        ],
        "source_stats": [s.model_dump() for s in r.source_stats],
    }


class ChatApp:
    """Holds the current report and agent; reloads on request."""

    def __init__(self, settings: Settings, *, use_cloud: bool = True, project_root: Path = PROJECT_ROOT) -> None:
        self.settings = settings
        self.use_cloud = use_cloud
        self.project_root = project_root
        self._lock = threading.Lock()
        self.snapshot: Snapshot | None = None
        self.agent: ReachAgent | None = None
        self.notes: list[str] = []

    @property
    def model(self) -> str:
        return self.settings.chat_model or self.settings.ollama_model

    def reload(self) -> None:
        snap, notes = load_latest(self.settings, self.project_root, use_cloud=self.use_cloud)
        if snap is None:
            notes = notes + [
                "No report yet. Run the pipeline (python -m agent_reach) or a cloud-runner workflow, then press Refresh."
            ]
        with self._lock:
            self.snapshot = snap
            self.notes = notes
            self.agent = ReachAgent(self.settings, snap) if snap else None
        if snap:
            log.info("chat: showing %s (%d trends, %.1f h old)", snap.origin_label, len(snap.report.macro_clusters), snap.age_hours)
        else:
            log.warning("chat: no report found (%s)", "; ".join(notes))

    def state(self) -> dict:
        with self._lock:
            return snapshot_json(self.snapshot, self.notes, self.model)


def make_handler(app: ChatApp, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ReachChat/1.0"

        def log_message(self, fmt: str, *args) -> None:  # route access logs through logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _host_ok(self) -> bool:
            # blocks DNS-rebinding pages from driving the local agent
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
            return False

        def _send(self, status: HTTPStatus, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            if self.path in ("/", "/index.html"):
                page = resources.files("agent_reach.chat").joinpath("page.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/snapshot":
                self._json(app.state())
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                # a cross-site form cannot send JSON without a CORS preflight, which this server never grants
                self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, b"expected application/json", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"too large", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(HTTPStatus.BAD_REQUEST, b"invalid JSON", "text/plain")
                return

            if self.path == "/api/refresh":
                app.reload()
                self._json(app.state())
            elif self.path == "/api/chat":
                self._chat(body)
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def _chat(self, body: dict) -> None:
            agent = app.agent
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()  # HTTP/1.0: the body streams until the connection closes
            events = (
                agent.respond(body.get("messages") or [])
                if agent
                else iter([{"type": "error", "text": " ".join(app.notes) or "No report loaded."}])
            )
            try:
                for ev in events:
                    self.wfile.write((json.dumps(ev) + "\n").encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log.debug("chat client disconnected mid-reply")

    return Handler


def serve(settings: Settings, *, port: int, use_cloud: bool, open_browser: bool) -> None:
    app = ChatApp(settings, use_cloud=use_cloud)
    log.info("chat: loading the latest report%s...", " (local + cloud)" if use_cloud else " (local only)")
    app.reload()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    log.info("chat: Reach is at %s (model %s). Ctrl+C to stop.", url, app.model)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent_reach chat", description="Chat with Reach about the latest trends report")
    p.add_argument("--port", type=int, help="local port (default AGENT_REACH_CHAT_PORT or 8765)")
    p.add_argument("--model", help="Ollama model for chat (default AGENT_REACH_CHAT_MODEL, else the labelling model)")
    p.add_argument("--no-cloud", action="store_true", help="only use local runs, never download cloud reports")
    p.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    p.add_argument("--db", type=Path, help="SQLite path (overrides AGENT_REACH_DB_PATH)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    from agent_reach.main import configure_logging

    configure_logging(args.log_level)
    settings = get_settings()
    updates: dict = {}
    if args.db:
        updates["db_path"] = args.db
    if args.model:
        updates["chat_model"] = args.model
    if updates:
        settings = settings.model_copy(update=updates)
    serve(settings, port=args.port or settings.chat_port, use_cloud=not args.no_cloud, open_browser=not args.no_browser)
    return 0
