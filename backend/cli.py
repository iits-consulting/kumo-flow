"""KumoFlow headless CLI — catalog / validate / run for external agents.

    uv run cli.py catalog [--input-kind image|video]
    uv run cli.py validate workflow.json [--target ID ...]
    uv run cli.py run workflow.json [--target ID ...] [--out DIR] [--local] [--run-id ID]
    uv run cli.py pull [file]              # live editor canvas -> file (default stdout)
    uv run cli.py push workflow.json       # replace the live canvas (retries once on rev conflict)

Accepts both the editor's workflow.json (serializeFlow) and deployment
artifacts (Graph.model_dump). `run` POSTs to the backend at $KUMOFLOW_API
(default http://localhost:8000) and falls back to an in-process run when the
server is unreachable.
"""

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from pydantic import ValidationError

from chat import EXCLUDED, _mate
from graph import Graph, _ancestors
from nodes import MEDIA, REGISTRY

API = os.environ.get("KUMOFLOW_API", "http://localhost:8000").rstrip("/")


def load_workflow(path: str) -> tuple[Graph, dict]:
    """Parse an editor workflow.json OR a deployment artifact into a Graph.
    Returns (graph, raw_dict). Raises ValueError with a one-line reason on
    malformed input."""
    try:
        raw = json.loads(Path(path).read_text())
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not JSON: {e}") from None
    return parse_workflow(raw, path), raw


def parse_workflow(raw: dict, src: str = "workflow") -> Graph:
    """A workflow dict in either shape (editor or deployment) -> Graph.
    The dict half of load_workflow, split out so the MCP server can validate
    dicts without touching disk. Raises ValueError, prefixed with src."""
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("edges"), list):
        raise ValueError(f"{src}: not a workflow file — expected {{nodes: [...], edges: [...]}}")
    nodes = []
    for n in raw["nodes"]:
        if isinstance(n, dict) and "data" in n:  # editor shape (serializeFlow)
            if n.get("type", "flow") != "flow":
                continue  # group/app nodes are visual only (mirrors toBackendPayload)
            nodes.append({"id": n.get("id"), "kind": n["data"].get("kind"), "config": n["data"].get("config") or {}})
        else:  # backend/deployment shape: already {id, kind, config}; extra keys (ui, label) ignored
            nodes.append(n)
    try:
        # NodeIn validation migrates KIND_ALIASES; Edge accepts sourceHandle/null
        return Graph(nodes=nodes, edges=raw["edges"], targets=raw.get("targets") or [])
    except ValidationError as e:
        first = e.errors()[0]
        raise ValueError(f"{src}: {'.'.join(map(str, first['loc']))}: {first['msg']}") from None


def default_targets(graph: Graph) -> list[str]:
    """Every Output-category node — what validate/run target when the file names none."""
    return [n.id for n in graph.nodes if (cls := REGISTRY.get(n.kind)) and cls.category == "Output"]


def catalog_specs(input_kind: str | None = None) -> list[dict]:
    """Node.spec() for every agent-visible kind — the authoritative node list
    (GET /nodes minus EXCLUDED), optionally filtered by modality."""
    return [
        cls.spec()
        for kind, cls in sorted(REGISTRY.items())
        if kind not in EXCLUDED and (not input_kind or input_kind in cls.modalities)
    ]


def to_editor_shape(raw: dict) -> dict:
    """Backend-shape nodes ({id, kind, config}) -> editor shape ({id, type:
    "flow", position, data}) so the canvas can render a pushed workflow. Data
    display fields come from Node.spec(); positions from a left-to-right
    auto-layout (x = depth*250, y = row*150). Editor-shape nodes and unknown
    kinds pass through untouched (validate reports the latter); non-workflow
    dicts too — the server's structural check rejects them."""
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("edges"), list):
        return raw
    # the canvas drops edges without an id (flow.ts:607) — stamp missing ones
    # with the editor's own convention (flow.ts:407)
    edges = [
        {"id": f"{e.get('source')}-{e.get('target')}-{e.get('sourceHandle') or ''}", **e}
        if isinstance(e, dict) and not e.get("id") else e
        for e in raw["edges"]
    ]
    bare = [n for n in raw["nodes"] if isinstance(n, dict) and "data" not in n and n.get("kind") in REGISTRY]
    if not bare:
        return {**raw, "edges": edges}
    # longest-path depth via Kahn's toposort — mirrors the editor's autoLayout
    ids = {n.get("id") for n in raw["nodes"] if isinstance(n, dict)}
    depth, indeg, out = {}, {}, {}
    for e in raw["edges"]:
        if isinstance(e, dict) and e.get("source") in ids and e.get("target") in ids:
            out.setdefault(e["source"], []).append(e["target"])
            indeg[e["target"]] = indeg.get(e["target"], 0) + 1
    queue = [i for i in ids if not indeg.get(i)]
    while queue:
        i = queue.pop()
        for t in out.get(i, ()):
            depth[t] = max(depth.get(t, 0), depth.get(i, 0) + 1)
            indeg[t] -= 1
            if indeg[t] == 0:
                queue.append(t)
    rows: dict[int, int] = {}  # nodes in a cycle keep depth 0 — cosmetic anyway

    def convert(n):
        spec = REGISTRY[n["kind"]].spec()
        data = {k: spec[k] for k in ("label", "kind", "color", "inputs", "outputs", "required_inputs", "config_info", "config_options", "config_hf")}
        data["config"] = {**spec["config"], **(n.get("config") or {})}  # defaults filled so the settings panel shows every field
        d = depth.get(n.get("id"), 0)
        rows[d] = rows.get(d, -1) + 1
        return {"id": n.get("id"), "type": "flow", "position": {"x": d * 250, "y": rows[d] * 150}, "data": data}

    return {**raw, "edges": edges, "nodes": [convert(n) if any(n is b for b in bare) else n for n in raw["nodes"]]}


def validate_graph(graph: Graph, targets: list[str]) -> tuple[list[str], list[str]]:
    """(errors, warnings). Empty errors = runnable. Whole-file sibling of
    chat.validate_plan — no chat-brief rules (one input, output_handling)."""
    errors, warnings = [], []
    by_id = {n.id: n for n in graph.nodes}

    for n in graph.nodes:
        cls = REGISTRY.get(n.kind)
        if cls is None:
            errors.append(f"node {n.id!r}: unknown kind {n.kind!r}")
            continue
        if n.kind == "custom":
            warnings.append(f"node {n.id!r}: custom node — unsandboxed exec; external agents must not author or modify these")
        try:
            cls(**n.config)  # same check _sigs/evaluate do
        except Exception as e:
            errors.append(f"node {n.id!r} ({n.kind}): bad config: {' '.join(str(e).split())}")

    wired: dict[tuple, int] = {}  # (target id, input port) -> edge count
    adj: dict[str, list] = {}  # edges between existing nodes, for the cycle walk
    for e in graph.edges:
        src, tgt = by_id.get(e.source), by_id.get(e.target)
        if src is None or tgt is None:
            errors.append(f"edge {e.source!r}->{e.target!r}: references a missing node")
            continue
        adj.setdefault(e.source, []).append(e.target)
        sp, tp = e.source_handle or "image", e.target_handle or "image"
        scls, tcls = REGISTRY.get(src.kind), REGISTRY.get(tgt.kind)
        if scls and src.kind != "custom" and sp not in scls.outputs:  # custom: dynamic ports
            errors.append(f"edge {e.source!r}->{e.target!r}: {src.kind!r} has no output port {sp!r}")
        if tcls and tgt.kind != "custom" and tp not in tcls.inputs:
            errors.append(f"edge {e.source!r}->{e.target!r}: {tgt.kind!r} has no input port {tp!r}")
        if not _mate(sp, tp):
            errors.append(f"edge {e.source!r}->{e.target!r}: output {sp!r} does not mate with input {tp!r}")
        wired[(e.target, tp)] = wired.get((e.target, tp), 0) + 1

    for (nid, port), count in wired.items():
        if count > 1 and port != "prompts":  # mirrors evaluate()'s port_value rule
            errors.append(f"node {nid!r}: input {port!r} has {count} connections (only 'prompts' merges)")

    for t in targets:
        if t not in by_id:
            errors.append(f"target {t!r} is not a node in the graph")

    # cycle — the visiting-set walk from _sigs, minus the hashing (and without
    # its blind spot: _sigs stops at uncacheable nodes before visiting inputs)
    visiting, done = set(), set()

    def cyclic(nid):
        if nid in visiting:
            return True
        if nid in done:
            return False
        visiting.add(nid)
        hit = any(cyclic(m) for m in adj.get(nid, ()))
        visiting.discard(nid)
        done.add(nid)
        return hit

    if any(cyclic(n.id) for n in graph.nodes):
        errors.append("graph has a cycle")

    # required port unwired: error on a target's ancestor path, warning off it
    upstream: dict[str, list] = {}
    for e in graph.edges:
        upstream.setdefault(e.target, []).append(e.source)
    on_path = set()
    for t in targets:
        if t in by_id:
            on_path |= _ancestors(t, upstream)
    for n in graph.nodes:
        cls = REGISTRY.get(n.kind)
        if cls is None:
            continue
        for p in cls.required_ports():
            if (n.id, p) not in wired:
                msg = f"node {n.id!r} ({n.kind}): required input {p!r} not connected"
                (errors if n.id in on_path else warnings).append(msg)
    return errors, warnings


def _http(method: str, path: str, payload: dict | None = None) -> dict:
    """One JSON request to the backend; raises urllib.error.HTTPError / OSError.
    The seam pull/push go through, so tests can swap in a TestClient."""
    req = urllib.request.Request(
        f"{API}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)


def _rewrite_media(value, fn):
    """Every '/media/...' string in a results tree -> fn(file name)."""
    if isinstance(value, str) and value.startswith("/media/"):
        return fn(Path(value).name)  # .name also blocks any traversal in server-sent names
    if isinstance(value, list):
        return [_rewrite_media(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_media(v, fn) for k, v in value.items()}
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("catalog", help="node specs as JSON — the authoritative node list")
    p.add_argument("--input-kind", choices=["image", "video"], help="only nodes supporting this modality")
    v = sub.add_parser("validate", help="check a workflow.json; exit 0 valid, 1 errors, 2 unparseable")
    r = sub.add_parser("run", help="run a workflow.json and print the results JSON")
    for p in (v, r):
        p.add_argument("workflow")
        p.add_argument("--target", action="append", default=[], help="target node id (repeatable); default: all Output nodes")
    r.add_argument("--out", help="download/copy result media into DIR, rewriting paths in the JSON")
    r.add_argument("--local", action="store_true", help="run in-process instead of via the server")
    r.add_argument("--run-id", default=None, help="run id for /progress and /stop (default: fresh uuid4)")
    pl = sub.add_parser("pull", help="fetch the live editor canvas as JSON; empty session prints a message and exits 1")
    pl.add_argument("file", nargs="?", help="write here instead of stdout")
    ps = sub.add_parser("push", help="replace the live editor canvas with a workflow.json (backend shape auto-converted to editor shape; retries once on rev conflict)")
    ps.add_argument("workflow")
    args = ap.parse_args(argv)

    if args.cmd == "catalog":
        print(json.dumps(catalog_specs(args.input_kind), indent=1, default=str))
        return 0

    if args.cmd == "pull":
        try:
            out = _http("GET", "/workflow")
        except OSError as e:
            print(f"error: backend unreachable at {API}: {getattr(e, 'reason', e)}")
            return 1
        if out["workflow"] is None:
            print("error: session is empty — no editor open and nothing pushed yet")
            return 1
        text = json.dumps(out["workflow"], indent=1)
        if args.file:
            Path(args.file).write_text(text + "\n")
        else:
            print(text)
        return 0

    if args.cmd == "push":
        try:
            raw = json.loads(Path(args.workflow).read_text())
        except OSError as e:
            print(f"error: cannot read {args.workflow}: {e}")
            return 2
        except json.JSONDecodeError as e:
            print(f"error: {args.workflow} is not JSON: {e}")
            return 2
        if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("edges"), list):
            print(f"error: {args.workflow}: not a workflow file — expected {{nodes: [...], edges: [...]}}")
            return 2
        raw = to_editor_shape(raw)  # the canvas only renders editor-shape nodes
        try:
            for attempt in (0, 1):  # GET rev, PUT; a 409 (concurrent write) gets one retry with a fresh rev
                rev = _http("GET", "/workflow")["rev"]
                try:
                    out = _http("PUT", "/workflow", {"workflow": raw, "rev": rev})
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 409 or attempt:
                        raise
        except urllib.error.HTTPError as e:
            print(f"error: PUT {API}/workflow -> {e.code}: {e.read().decode(errors='replace')}")
            return 1
        except OSError as e:
            print(f"error: backend unreachable at {API}: {getattr(e, 'reason', e)}")
            return 1
        print(json.dumps(out))
        return 0

    try:
        graph, _ = load_workflow(args.workflow)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    targets = args.target or graph.targets or default_targets(graph)
    if not targets:
        print("error: no output nodes — pass --target")
        return 1

    if args.cmd == "validate":
        errors, warnings = validate_graph(graph, targets)
        for e in errors:
            print(f"error: {e}")
        for w in warnings:
            print(f"warning: {w}")
        return 1 if errors else 0

    # run — server first (warm models, shared cache), local as fallback
    graph.targets = list(dict.fromkeys(targets))
    graph.run_id = args.run_id or uuid.uuid4().hex
    local, out = args.local, None
    if local:
        print("run: local (--local)", file=sys.stderr)
    else:
        req = urllib.request.Request(
            f"{API}/run",
            data=json.dumps(graph.model_dump(by_alias=True)).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=float(os.environ.get("KUMOFLOW_RUN_TIMEOUT", "3600"))) as res:
                out = json.load(res)
            print(f"run: server ({API})", file=sys.stderr)
        except urllib.error.HTTPError as e:  # the server answered — a real failure, don't rerun locally
            print(json.dumps({"error": f"POST {API}/run -> {e.code}: {e.read().decode(errors='replace')}"}))
            return 1
        except OSError as e:  # unreachable / refused / DNS
            print(f"run: local (server unreachable at {API}: {getattr(e, 'reason', e)})", file=sys.stderr)
            local = True
    if out is None:
        import main  # lazy: fastapi + chat only ever load on the local path

        try:
            out = main.run_graph(graph)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return 1

    if args.out:
        dest_dir = Path(args.out)
        dest_dir.mkdir(parents=True, exist_ok=True)

        def fetch(name):
            dest = dest_dir / name
            if not dest.exists():  # media names are content-addressed
                if local:
                    shutil.copyfile(MEDIA / name, dest)
                else:
                    with urllib.request.urlopen(f"{API}/media/{name}") as r:
                        dest.write_bytes(r.read())
            return str(dest)

        out = _rewrite_media(out, fetch)
    elif local:  # /media URLs mean nothing without a server — hand back real paths
        out = _rewrite_media(out, lambda name: str(MEDIA / name))
    print(json.dumps(out, indent=1))
    return 1 if any(isinstance(r, dict) and "error" in r for r in out.get("results", {}).values()) else 0


if __name__ == "__main__":
    sys.exit(main())
