"""KumoFlow MCP server — the CLI's surface as eight tools, stdio transport.

    uv run --extra agent mcp_server.py

Agent setup:
    claude mcp add kumoflow -- uv run --directory <repo>/backend --extra agent mcp_server.py

Never imports main.py, never loads models in-process — one stdio child runs
per agent session, so every execution tool delegates to the backend HTTP API
at $KUMOFLOW_API. Graph logic is imported from cli.py, zero duplication.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

try:  # mcp >= 2.0 renamed FastMCP -> MCPServer
    from mcp.server import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from cli import API, catalog_specs, default_targets, parse_workflow, to_editor_shape, validate_graph
from graph import Graph


def _instructions() -> str:
    """AGENTS.md minus its CLI and MCP-setup sections — the workflow schema,
    port contract and hard rules every connected agent needs (the same
    guidance chat.py bakes into the local LLM's system prompt). Clients like
    Claude Code inject this into the agent's system prompt."""
    text = (Path(__file__).resolve().parent.parent / "AGENTS.md").read_text()
    return "\n## ".join(s for s in text.split("\n## ") if not s.startswith(("CLI\n", "MCP server\n")))


mcp = FastMCP("kumoflow", instructions=_instructions())


def _api(path: str, payload: dict | None = None, timeout: float = 30.0, raw: bool = False, method: str | None = None):
    """GET (payload None) or POST JSON to the backend (override with method=).
    Down backend -> one clear line, not a traceback (FastMCP relays the
    exception message)."""
    req = urllib.request.Request(
        f"{API}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read() if raw else json.load(res)
    except urllib.error.HTTPError as e:  # the server answered — a real failure
        err = RuntimeError(f"{req.get_method()} {API}{path} -> {e.code}: {e.read().decode(errors='replace')}")
        err.code = e.code  # so set_canvas can branch on 409 without parsing the message
        raise err from None
    except OSError:  # unreachable / refused / DNS / timeout
        raise RuntimeError(f"KumoFlow backend not reachable at {API} — start it with `make up`") from None


def _targets(graph: Graph, targets: list[str] | None) -> list[str]:
    tgts = list(dict.fromkeys(targets or graph.targets or default_targets(graph)))
    if not tgts:
        raise ValueError("no output nodes — pass targets")
    return tgts


@mcp.tool()
def catalog(input_kind: str | None = None) -> list[dict]:
    """Node specs for every available kind — the authoritative node list.
    Optionally filter by modality: 'image' or 'video'."""
    return catalog_specs(input_kind)


@mcp.tool()
def validate_workflow(workflow: dict, targets: list[str] | None = None) -> dict:
    """Validate a workflow dict (editor or deployment shape) against the given
    targets (default: its Output nodes). Empty errors = runnable."""
    graph = parse_workflow(workflow)
    errors, warnings = validate_graph(graph, _targets(graph, targets))
    return {"errors": errors, "warnings": warnings}


@mcp.tool()
def run_workflow(workflow: dict, targets: list[str] | None = None, run_id: str | None = None) -> dict:
    """Run a workflow on the backend; blocks until done (runs legitimately take
    minutes — poll get_progress from another turn). Returns {run_id, results};
    result media are '/media/...' URLs for fetch_media."""
    graph = parse_workflow(workflow)
    graph.targets = _targets(graph, targets)
    graph.run_id = run_id or uuid.uuid4().hex
    out = _api("/run", graph.model_dump(by_alias=True), timeout=float(os.environ.get("KUMOFLOW_RUN_TIMEOUT", "3600")))
    return {"run_id": graph.run_id, "results": out["results"]}


@mcp.tool()
def get_progress(run_id: str) -> dict:
    """Progress of a running run: {nodes: {node_id: [done, total]}} — empty
    once it finishes."""
    return _api(f"/progress/{urllib.parse.quote(run_id, safe='')}")


@mcp.tool()
def stop_run(run_id: str, target: str | None = None) -> dict:
    """Stop a run — whole run, or just one target's subgraph when target is
    given. Returns {stopped: bool} (False = no such live run)."""
    q = f"?target={urllib.parse.quote(target)}" if target else ""
    out = _api(f"/stop/{urllib.parse.quote(run_id, safe='')}{q}", payload={})
    return {"stopped": out["stopping"]}


@mcp.tool()
def fetch_media(media_url: str, out_path: str) -> dict:
    """Save a result's '/media/...' URL to out_path so the agent can Read the
    file. Returns {path} — media as paths, never base64 blobs."""
    name = Path(media_url).name  # .name also blocks traversal (mirrors cli._rewrite_media)
    data = _api(f"/media/{urllib.parse.quote(name)}", raw=True)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"path": str(dest)}


@mcp.tool()
def get_canvas() -> dict:
    """The live editor canvas: {rev, workflow} — workflow is editor-shape JSON
    (nodes {id, type, position, data: {kind, config, ...}}), or null when no
    canvas is open and nothing was pushed yet."""
    return _api("/workflow")


@mcp.tool()
def set_canvas(workflow: dict, rev: int | None = None) -> dict:
    """Replace the live editor canvas with workflow — appears in the open
    browser within ~2 s. Accepts editor-shape or backend-shape workflows:
    backend-shape nodes ({id, kind, config}) are auto-converted to editor shape
    with auto-layout; editor-shape nodes (minimal template:
    {id, type: "flow", position: {x, y}, data: {kind, config, label, color,
    inputs, outputs, required_inputs}}) are stored verbatim. rev None = use the
    current rev, retrying once if a concurrent write lands; an explicit rev
    fails fast on conflict. Rejects workflows validate_workflow would flag
    with errors (warnings don't block, and a draft without output nodes is
    allowed). Returns {rev}."""
    graph = parse_workflow(workflow)
    errors, _ = validate_graph(graph, graph.targets or default_targets(graph))
    if errors:
        raise ValueError("invalid workflow — fix and retry: " + "; ".join(errors))
    workflow = to_editor_shape(workflow)
    for attempt in (0, 1):
        r = _api("/workflow")["rev"] if rev is None else rev
        try:
            return _api("/workflow", {"workflow": workflow, "rev": r}, method="PUT")
        except RuntimeError as e:
            if rev is not None or attempt or getattr(e, "code", None) != 409:
                raise


if __name__ == "__main__":
    mcp.run()  # stdio — HTTP/SSE transport is explicitly out of scope in v1
