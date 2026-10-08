"""Tests for the headless CLI (cli.py): loader, validator, run --local.

    uv run pytest test_cli.py
"""

import base64
import json
from pathlib import Path

import pytest
import torch
from torchvision.io import encode_png

import cli
import graph
from graph import Graph


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The cross-run result cache is process-global — isolate every test."""
    graph.CACHE.clear()
    yield
    graph.CACHE.clear()


def _png_data_url(chw_uint8):
    return "data:image/png;base64," + base64.b64encode(encode_png(chw_uint8).numpy().tobytes()).decode()


def _flow_node(nid, kind, config=None):
    return {"id": nid, "type": "flow", "position": {"x": 0, "y": 0}, "data": {"label": kind, "kind": kind, "config": config or {}}}


def _editor_file(tmp_path, data_url=""):
    """load -> flip -> view in editor (serializeFlow) shape, plus visual-only nodes."""
    wf = {
        "nodes": [
            _flow_node("load-1", "load", {"data": data_url}),
            _flow_node("flip-2", "flip"),
            _flow_node("view-3", "view"),
            {"id": "group-1", "type": "group", "position": {"x": 0, "y": 0}, "data": {"label": "box"}},
            {"id": "app-1", "type": "app", "position": {"x": 0, "y": 0}, "data": {"title": "App", "sections": []}},
        ],
        "edges": [
            {"id": "e1", "source": "load-1", "target": "flip-2", "sourceHandle": None, "targetHandle": None},
            {"id": "e2", "source": "flip-2", "target": "view-3", "sourceHandle": "image", "targetHandle": "image"},
        ],
    }
    p = tmp_path / "workflow.json"
    p.write_text(json.dumps(wf))
    return p


# --- loader -------------------------------------------------------------------


def test_load_editor_shape(tmp_path):
    url = _png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8))
    g, raw = cli.load_workflow(str(_editor_file(tmp_path, url)))
    assert [n.kind for n in g.nodes] == ["load", "flip", "view"]  # group/app dropped
    assert g.nodes[0].config == {"data": url}
    assert len(g.edges) == 2 and g.targets == []
    assert g.edges[0].source_handle is None  # null handles pass through (evaluate defaults them)
    assert len(raw["nodes"]) == 5  # raw dict untouched


def test_load_deployment_shape(tmp_path):
    spec = {
        "nodes": [{"id": "l", "kind": "load", "config": {}}, {"id": "v", "kind": "view", "config": {}}],
        "edges": [{"source": "l", "target": "v", "sourceHandle": "image", "targetHandle": "image"}],
        "targets": ["v"],
        "ui": {"title": "x"},  # ignored
    }
    p = tmp_path / "deploy.json"
    p.write_text(json.dumps(spec))
    g, _ = cli.load_workflow(str(p))
    assert [n.kind for n in g.nodes] == ["load", "view"] and g.targets == ["v"]


def test_load_kind_alias_migrates(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"nodes": [{"id": "s", "kind": "sam3"}], "edges": []}))
    g, _ = cli.load_workflow(str(p))
    assert g.nodes[0].kind == "segment"
    assert g.nodes[0].config["model_id"] == "facebook/sam3"


def test_load_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(ValueError, match="not JSON"):
        cli.load_workflow(str(p))
    p.write_text(json.dumps({"nodes": {}}))
    with pytest.raises(ValueError, match="expected"):
        cli.load_workflow(str(p))
    with pytest.raises(ValueError, match="cannot read"):
        cli.load_workflow(str(tmp_path / "missing.json"))


# --- validator ------------------------------------------------------------------


def _g(nodes, edges=(), targets=()):
    """nodes: (id, kind, config) triples; edges: (src, tgt, src_port, tgt_port)."""
    return Graph(
        nodes=[{"id": i, "kind": k, "config": c} for i, k, c in nodes],
        edges=[{"source": s, "target": t, "sourceHandle": sp, "targetHandle": tp} for s, t, sp, tp in edges],
        targets=list(targets),
    )


def test_validate_unknown_kind():
    errs, _ = cli.validate_graph(_g([("x", "nope", {})]), [])
    assert errs == ["node 'x': unknown kind 'nope'"]


def test_validate_bad_config():
    errs, _ = cli.validate_graph(_g([("lv", "load_video", {"fps": {"a": 1}})]), [])
    assert len(errs) == 1 and "bad config" in errs[0]


def test_validate_bad_mate():
    g = _g([("l", "load", {}), ("s", "segment", {})], [("l", "s", "image", "prompts")])
    errs, _ = cli.validate_graph(g, [])
    assert errs == ["edge 'l'->'s': output 'image' does not mate with input 'prompts'"]


def test_validate_unknown_port():
    g = _g([("l", "load", {}), ("v", "view", {})], [("l", "v", "bogus", "image")])
    errs, _ = cli.validate_graph(g, [])
    assert any("has no output port 'bogus'" in e for e in errs)


def test_validate_double_wire_only_prompts_merges():
    g = _g(
        [("l1", "load", {}), ("l2", "load", {}), ("v", "view", {})],
        [("l1", "v", "image", "image"), ("l2", "v", "image", "image")],
    )
    errs, _ = cli.validate_graph(g, [])
    assert any("only 'prompts' merges" in e for e in errs)
    g = _g(
        [("t1", "text_prompt", {"text": "a"}), ("t2", "text_prompt", {"text": "b"}), ("s", "segment", {})],
        [("t1", "s", "prompts", "prompts"), ("t2", "s", "prompts", "prompts")],
    )
    assert cli.validate_graph(g, []) == ([], [])


def test_validate_cycle():
    g = _g([("f1", "flip", {}), ("f2", "flip", {})], [("f1", "f2", "image", "image"), ("f2", "f1", "image", "image")])
    errs, _ = cli.validate_graph(g, [])
    assert "graph has a cycle" in errs


def test_validate_edge_to_missing_node():
    errs, _ = cli.validate_graph(_g([("l", "load", {})], [("l", "ghost", "image", "image")]), [])
    assert errs == ["edge 'l'->'ghost': references a missing node"]


def test_validate_bad_target():
    errs, _ = cli.validate_graph(_g([("l", "load", {})]), ["nope"])
    assert "target 'nope' is not a node in the graph" in errs


def test_validate_required_port_on_path_error_off_path_warning():
    nodes = [("l", "load", {}), ("v1", "view", {}), ("v2", "view", {})]
    edges = [("l", "v1", "image", "image")]  # v2 is a half-built branch
    errs, warns = cli.validate_graph(_g(nodes, edges), ["v1"])
    assert errs == [] and warns == ["node 'v2' (view): required input 'image' not connected"]
    errs, _ = cli.validate_graph(_g(nodes, edges), ["v2"])  # now it's on-path
    assert "node 'v2' (view): required input 'image' not connected" in errs


def test_validate_custom_warns_note_does_not():
    _, warns = cli.validate_graph(_g([("c", "custom", {})]), [])
    assert any("unsandboxed" in w for w in warns)
    assert cli.validate_graph(_g([("n", "note", {})]), []) == ([], [])


# --- CLI commands -----------------------------------------------------------------


def test_catalog_excludes_and_filters(capsys):
    assert cli.main(["catalog"]) == 0
    kinds = {s["kind"] for s in json.loads(capsys.readouterr().out)}
    assert "load" in kinds and "custom" not in kinds and "note" not in kinds
    assert cli.main(["catalog", "--input-kind", "video"]) == 0
    kinds = {s["kind"] for s in json.loads(capsys.readouterr().out)}
    assert "load_video" in kinds and "load" not in kinds  # load is image-only


def test_validate_exit_codes(tmp_path, capsys):
    assert cli.main(["validate", str(_editor_file(tmp_path))]) == 0  # default target: the view node
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nodes": [{"id": "x", "kind": "nope"}], "edges": [], "targets": ["x"]}))
    assert cli.main(["validate", str(bad)]) == 1
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{")
    assert cli.main(["validate", str(garbage)]) == 2
    out = capsys.readouterr().out
    assert "error: node 'x': unknown kind 'nope'" in out


def test_validate_no_targets(tmp_path, capsys):
    p = tmp_path / "no_out.json"
    p.write_text(json.dumps({"nodes": [{"id": "l", "kind": "load"}], "edges": []}))
    assert cli.main(["validate", str(p)]) == 1
    assert "no output nodes — pass --target" in capsys.readouterr().out


# --- canvas session (/workflow + pull/push) -----------------------------------------
# main is a heavy import but already loads on the run --local path above; the
# TestClient is not entered as a context manager, so the lifespan (Ollama) never runs.


@pytest.fixture()
def canvas_client(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setitem(main.SESSION, "workflow", None)
    monkeypatch.setitem(main.SESSION, "rev", 0)
    return TestClient(main.app)


def _wire_cli_to(client, monkeypatch):
    """Route cli._http through the TestClient instead of a live server."""
    import io
    import urllib.error

    def fake(method, path, payload=None):
        res = client.request(method, path, json=payload)
        if res.status_code >= 400:
            raise urllib.error.HTTPError(path, res.status_code, "", None, io.BytesIO(res.content))
        return res.json()

    monkeypatch.setattr(cli, "_http", fake)
    return fake


def test_workflow_roundtrip(canvas_client):
    assert canvas_client.get("/workflow").json() == {"workflow": None, "rev": 0}
    wf = {"nodes": [], "edges": []}
    res = canvas_client.put("/workflow", json={"workflow": wf, "rev": 0})
    assert res.status_code == 200 and res.json() == {"rev": 1}
    assert canvas_client.get("/workflow").json() == {"workflow": wf, "rev": 1}
    assert canvas_client.put("/workflow", json={"workflow": {"nodes": {}}, "rev": 1}).status_code == 422  # structure only


def test_workflow_stale_rev_409_carries_current_state(canvas_client):
    wf = {"nodes": [], "edges": [], "who": "first"}
    canvas_client.put("/workflow", json={"workflow": wf, "rev": 0})
    res = canvas_client.put("/workflow", json={"workflow": {"nodes": [], "edges": []}, "rev": 0})  # stale
    assert res.status_code == 409
    assert res.json() == {"workflow": wf, "rev": 1}  # loser rebases without a second GET


def test_cli_push_retries_once_on_409(canvas_client, tmp_path, capsys, monkeypatch):
    import main

    real = _wire_cli_to(canvas_client, monkeypatch)
    puts = []

    def racy(method, path, payload=None):
        if method == "PUT" and not puts:  # a concurrent writer lands between GET rev and the first PUT
            puts.append(1)
            canvas_client.put("/workflow", json={"workflow": {"nodes": [], "edges": []}, "rev": main.SESSION["rev"]})
        return real(method, path, payload)

    monkeypatch.setattr(cli, "_http", racy)
    p = tmp_path / "wf.json"
    p.write_text(json.dumps({"nodes": [], "edges": [], "who": "cli"}))
    assert cli.main(["push", str(p)]) == 0
    assert json.loads(capsys.readouterr().out) == {"rev": 2}
    assert main.SESSION["workflow"]["who"] == "cli"  # the retry won

    # a second conflict is not retried again: every PUT loses the race -> exit 1
    def always_racy(method, path, payload=None):
        if method == "PUT":
            canvas_client.put("/workflow", json={"workflow": {"nodes": [], "edges": []}, "rev": main.SESSION["rev"]})
        return real(method, path, payload)

    monkeypatch.setattr(cli, "_http", always_racy)
    assert cli.main(["push", str(p)]) == 1
    assert "409" in capsys.readouterr().out


def test_cli_push_bad_file(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"nodes": {}}))
    assert cli.main(["push", str(p)]) == 2
    assert "expected" in capsys.readouterr().out


def test_to_editor_shape_converts_backend_nodes():
    raw = {
        "nodes": [
            {"id": "l1", "kind": "load", "config": {}},
            {"id": "l2", "kind": "load", "config": {}},
            {"id": "b", "kind": "blur", "config": {"blur": 5}},
            {"id": "v", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "l1", "target": "b"}, {"source": "b", "target": "v"}],
    }
    out = cli.to_editor_shape(raw)
    nodes = {n["id"]: n for n in out["nodes"]}
    assert all(n["type"] == "flow" for n in nodes.values())
    pos = {i: n["position"] for i, n in nodes.items()}
    assert pos["l1"]["x"] < pos["b"]["x"] < pos["v"]["x"]  # x follows depth
    assert pos["l1"] != pos["l2"]  # same column, distinct rows
    assert nodes["b"]["data"]["kind"] == "blur" and nodes["b"]["data"]["config"]["blur"] == 5
    from nodes import REGISTRY

    spec = REGISTRY["blur"].spec()
    for k in ("label", "color", "inputs", "outputs", "required_inputs", "config_info", "config_options"):
        assert nodes["b"]["data"][k] == spec[k]
    # edge content preserved, missing ids stamped (the canvas drops id-less edges)
    assert [{k: v for k, v in e.items() if k != "id"} for e in out["edges"]] == raw["edges"]
    ids = [e["id"] for e in out["edges"]]
    assert all(ids) and len(set(ids)) == len(ids)


def test_to_editor_shape_passthrough(tmp_path):
    wf = json.loads(_editor_file(tmp_path).read_text())
    assert cli.to_editor_shape(wf) == wf  # editor shape is left byte-identical


def test_to_editor_shape_unknown_kind_left_alone():
    unknown = {"nodes": [{"id": "x", "kind": "nope"}], "edges": []}
    assert cli.to_editor_shape(unknown) == unknown  # validate's job, not a crash


def test_cli_push_converts_backend_shape(canvas_client, tmp_path, capsys, monkeypatch):
    import main

    _wire_cli_to(canvas_client, monkeypatch)
    p = tmp_path / "wf.json"
    p.write_text(json.dumps({"nodes": [{"id": "l", "kind": "load", "config": {}}], "edges": []}))
    assert cli.main(["push", str(p)]) == 0
    stored = main.SESSION["workflow"]["nodes"][0]
    assert stored["type"] == "flow" and "position" in stored and stored["data"]["kind"] == "load"


def test_run_local(tmp_path, capsys):
    wf = _editor_file(tmp_path, _png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8)))
    out_dir = tmp_path / "out"
    assert cli.main(["run", str(wf), "--local", "--out", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert "run: local (--local)" in captured.err
    res = json.loads(captured.out)["results"]["view-3"]
    assert res["sizes"] == [[5, 4]]  # [W, H] of the source image
    assert len(res["images"]) == 1
    img = Path(res["images"][0])
    assert img.is_file() and img.parent == out_dir  # media copied out, JSON rewritten
