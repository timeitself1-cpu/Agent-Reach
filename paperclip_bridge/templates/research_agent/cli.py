"""File-in/file-out entry point for the LangGraph research workflow.

    python -m research_agent.cli --in request.json --out result.json

Knows nothing about Paperclip. It reads a request (schema_version 1.0), runs
the LangGraph graph once and writes a result:

    request: {"schema_version", "request_id", "title", "instructions", "created_at"}
    result:  {"schema_version", "request_id", "status": "ok"|"no_results"|"error",
              "summary", "papers": [...], "handoff": {"title", "instructions"} | null, "error"}

Exit codes: 0 = result written (research failures are status "error" inside
the file), 2 = unreadable request, 3 = result could not be written.

Settings (environment):
    RESEARCH_AGENT_GRAPH             "module:attribute" of the graph. The attribute may be
                                     a compiled graph, an uncompiled StateGraph, or a
                                     zero-argument factory returning either.
                                     Default: research_agent.graph:graph
    RESEARCH_AGENT_RECURSION_LIMIT   LangGraph recursion limit (default 50)
    RESEARCH_AGENT_AUTO_HANDOFF      1 = build a coder hand-off from the best paper when the
                                     graph returns none (default 1)
    RESEARCH_AGENT_HANDOFF_MIN_CONFIDENCE  minimum paper confidence for an automatic hand-off
                                     (default 0.7)

Adapt ``build_input_state`` and ``extract_output`` if your graph state uses
other keys. Everything else is fixed by the file contract.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_GRAPH = "research_agent.graph:graph"


# ----------------------------------------------------------- state mapping
def build_input_state(request: dict[str, Any]) -> dict[str, Any]:
    """Request -> initial LangGraph state."""
    return {
        "topic": request["title"],
        "instructions": request.get("instructions", ""),
        "request_id": request["request_id"],
    }


def extract_output(state: dict[str, Any]) -> tuple[list[Any], str, Any]:
    """Final LangGraph state -> (papers, summary, handoff)."""
    papers = state.get("papers") or state.get("results") or []
    summary = state.get("summary") or state.get("synthesis") or state.get("final_report") or ""
    return list(papers), _as_text(summary), state.get("handoff")


# ----------------------------------------------------------- normalisation
def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"cannot convert {type(obj).__name__} to a mapping")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    content = getattr(value, "content", None)  # e.g. an AIMessage
    if isinstance(content, str):
        return content.strip()
    return str(value).strip()


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [_as_text(v) for v in value if _as_text(v)]


def normalize_paper(raw: Any) -> dict[str, Any] | None:
    p = _as_dict(raw)
    title = _as_text(p.get("title"))
    summary = _as_text(p.get("summary") or p.get("synthesis") or p.get("abstract"))
    if not title or not summary:
        return None
    try:
        confidence = float(p.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "title": title,
        "url": _as_text(p.get("url") or p.get("link")),
        "arxiv_id": _as_text(p.get("arxiv_id")),
        "summary": summary,
        "key_claims": _str_list(p.get("key_claims")),
        "methods": _as_text(p.get("methods")),
        "implementation_notes": _as_text(p.get("implementation_notes")),
        "code_links": _str_list(p.get("code_links")),
        "confidence": min(1.0, max(0.0, confidence)),
    }


def normalize_handoff(raw: Any, papers: list[dict[str, Any]]) -> dict[str, str] | None:
    if raw:
        h = _as_dict(raw)
        title = _as_text(h.get("title"))[:200]
        if title:
            return {"title": title, "instructions": _as_text(h.get("instructions") or h.get("description"))}
    if os.environ.get("RESEARCH_AGENT_AUTO_HANDOFF", "1") != "1":
        return None
    min_conf = float(os.environ.get("RESEARCH_AGENT_HANDOFF_MIN_CONFIDENCE", "0.7"))
    buildable = [p for p in papers if p["implementation_notes"] and p["confidence"] >= min_conf]
    if not buildable:
        return None
    best = max(buildable, key=lambda p: p["confidence"])
    lines = [f"Prototype the approach from \"{best['title']}\".", "", best["implementation_notes"]]
    if best["code_links"]:
        lines += ["", "Reference code: " + ", ".join(best["code_links"])]
    if best["url"]:
        lines += ["", f"Paper: {best['url']}"]
    return {"title": f"Prototype: {best['title']}"[:200], "instructions": "\n".join(lines)}


# ----------------------------------------------------------- graph
def load_graph(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"RESEARCH_AGENT_GRAPH must look like 'package.module:attribute', got {spec!r}")
    obj = getattr(importlib.import_module(module_name), attr)
    if not hasattr(obj, "invoke") and not hasattr(obj, "compile") and callable(obj):
        obj = obj()
    if not hasattr(obj, "invoke") and hasattr(obj, "compile"):
        obj = obj.compile()
    if not hasattr(obj, "invoke"):
        raise TypeError(f"{spec} is not a LangGraph graph (no .invoke)")
    return obj


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Run the graph and build a result. Never raises: failures become status 'error'."""
    base = {"schema_version": SCHEMA_VERSION, "request_id": request["request_id"]}
    try:
        graph = load_graph(os.environ.get("RESEARCH_AGENT_GRAPH", DEFAULT_GRAPH))
        limit = int(os.environ.get("RESEARCH_AGENT_RECURSION_LIMIT", "50"))
        state = graph.invoke(build_input_state(request), config={"recursion_limit": limit})
        raw_papers, summary, raw_handoff = extract_output(_as_dict(state))
        papers = [p for p in (normalize_paper(r) for r in raw_papers) if p]
        handoff = normalize_handoff(raw_handoff, papers)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {**base, "status": "error", "summary": "", "papers": [], "handoff": None,
                "error": f"{type(exc).__name__}: {exc}"[:2000]}
    return {
        **base,
        "status": "ok" if papers else "no_results",
        "summary": summary or f"{len(papers)} paper(s) synthesised.",
        "papers": papers,
        "handoff": handoff if papers else None,
        "error": "",
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research_agent.cli")
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        request = json.loads(args.inp.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or not request.get("request_id") or not request.get("title"):
            raise ValueError("request needs 'request_id' and 'title'")
    except (OSError, ValueError) as exc:
        print(f"research_agent.cli: bad request file {args.inp}: {exc}", file=sys.stderr)
        return 2
    result = run(request)
    try:
        write_atomic(args.out, result)
    except OSError as exc:
        print(f"research_agent.cli: cannot write {args.out}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"status": result["status"], "papers": len(result["papers"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
