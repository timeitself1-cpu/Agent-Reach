# paperclip_bridge

Runs a Paperclip-unaware research CLI (for example `research_agent`) from a
Paperclip heartbeat. Paperclip handles scheduling and tickets. The CLI does
the research and only ever reads one JSON file and writes another.

```
Paperclip heartbeat (process adapter)
  └─ python -m paperclip_bridge
       1. GET  /api/agents/me/inbox-lite          pick a claimable ticket
       2. take the machine-wide GPU lock          busy -> exit 0, ticket untouched
       3. POST /api/issues/{id}/checkout          409 -> try the next ticket
       4. write request.json, run  <cmd> --in request.json --out result.json
       5. release the GPU lock
       6. PUT  /api/issues/{id}/documents/research-result   full result (markdown + JSON)
          POST /api/companies/{cid}/issues        hand-off ticket for the coder agent (once)
          PATCH /api/issues/{id}                  done + summary comment, or blocked + reason
       7. print {"summary": ...} and exit 0
```

The bridge exits non-zero only when it fails itself (for example, Paperclip is
unreachable). A research failure becomes a `blocked` ticket with the reason and
a stderr tail in a comment, so it shows up on the board.

## File contract (`contract.py`, `schema_version` 1.0)

`python -m paperclip_bridge schema` prints the JSON Schema of both files.

`request.json` (written by the bridge):

```json
{
  "schema_version": "1.0",
  "request_id": "RES-12-run-uuid",
  "title": "Summarise this week's agent papers",
  "instructions": "Focus on tool use. Sources: dair-ai/AI-Papers-of-the-Week, arXiv cs.CL.",
  "created_at": "2026-09-25T09:00:00Z"
}
```

`result.json` (written by the CLI):

```json
{
  "schema_version": "1.0",
  "request_id": "RES-12-run-uuid",
  "status": "ok",
  "summary": "Three agent papers this week; one is worth prototyping.",
  "papers": [{
    "title": "...", "url": "https://arxiv.org/abs/...", "arxiv_id": "...",
    "summary": "...", "key_claims": ["..."], "methods": "...",
    "implementation_notes": "What a coder would need to build it.",
    "code_links": ["https://github.com/..."], "confidence": 0.8
  }],
  "handoff": {"title": "Prototype X", "instructions": "..."},
  "error": ""
}
```

* `status`: `ok` or `no_results` marks the ticket `done`. `error` marks it
  `blocked` and uses `error` as the reason.
* `request_id` must echo the request's value.
* `handoff` is optional. When it is set and `RESEARCH_BRIDGE_HANDOFF_AGENT_ID`
  is configured, the bridge creates one child ticket (`parentId` = research
  ticket, same goal and project) assigned to that agent. Assignment wakes the
  agent.
* Exit `0` means `result.json` was written. Any other exit code, a timeout, or
  a missing or invalid file marks the ticket `blocked`.

The CLI side needs only a thin wrapper around your existing code:

```python
# research_agent/__main__.py
import argparse, json

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    req = json.load(open(a.inp, encoding="utf-8"))
    try:
        papers, summary, handoff = run_research(req["title"], req["instructions"])  # your LangGraph graph
        result = {"request_id": req["request_id"], "status": "ok" if papers else "no_results",
                  "summary": summary, "papers": papers, "handoff": handoff}
    except Exception as exc:
        result = {"request_id": req["request_id"], "status": "error", "error": str(exc)}
    json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)
    return 0

raise SystemExit(main())
```

## Paperclip agent setup

Adapter `process`:

```json
{
  "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
  "args": ["-m", "paperclip_bridge"],
  "cwd": "C:\\path\\to\\toolkit",
  "timeoutSec": 4200,
  "env": {
    "RESEARCH_BRIDGE_CMD": "C:\\path\\to\\venv\\Scripts\\python.exe -m research_agent",
    "RESEARCH_BRIDGE_HANDOFF_AGENT_ID": "<coder agent uuid>"
  }
}
```

Keep `timeoutSec` above `RESEARCH_BRIDGE_TIMEOUT_S` plus `RESEARCH_BRIDGE_GPU_WAIT_S`
so the bridge can report a CLI timeout before Paperclip kills the bridge.

Agent `runtimeConfig.heartbeat`:

```json
{ "enabled": true, "intervalSec": 900, "maxConcurrentRuns": 1, "wakeOnDemand": true }
```

* `maxConcurrentRuns: 1` matters: Paperclip's default is 20 concurrent runs
  per agent.
* `wakeOnDemand` starts a run when a ticket is assigned. `intervalSec` is the
  retry path for tickets skipped because the GPU was busy.

## VRAM (12 GB)

* Every process that loads a local model takes the same lock file
  (`RESEARCH_BRIDGE_GPU_LOCK_PATH`, default `~/.paperclip_bridge/gpu.lock`).
  Other Python workers can use `paperclip_bridge.gpu_lock.GpuLock` directly.
* Ollama: `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`.
* Do not keep a model loaded in LM Studio while Ollama is serving. Neither
  can see the other's VRAM use.

## Settings (`RESEARCH_BRIDGE_*`)

| Variable | Default | Meaning |
|---|---|---|
| `CMD` | `python -m research_agent` | Research CLI; `--in`/`--out` are appended |
| `CWD` | bridge cwd | Working directory for the CLI |
| `WORKDIR` | `~/.paperclip_bridge/runs` | Request/result/log files per ticket and run, plus hand-off markers |
| `TIMEOUT_S` | `3600` | CLI time limit |
| `GPU_LOCK_PATH` | `~/.paperclip_bridge/gpu.lock` | Shared lock file |
| `GPU_WAIT_S` | `30` | How long to wait for the lock before leaving tickets for the next heartbeat |
| `HANDOFF_AGENT_ID` | empty | Agent that receives hand-off tickets; empty disables them |
| `DOCUMENT_KEY` | `research-result` | Issue document key for the full result |
| `API_TIMEOUT_S` | `30` | Paperclip API timeout |

Paperclip injects `PAPERCLIP_API_URL`, `PAPERCLIP_API_KEY`, `PAPERCLIP_RUN_ID`,
`PAPERCLIP_AGENT_ID` and `PAPERCLIP_COMPANY_ID`. The bridge refuses to run
without them.
