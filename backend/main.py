"""KumoFlow backend — serves the node palette and runs a flow graph up to a target node.

Run the server:  uv run main.py
Run the tests:   uv run pytest   (see test_nodes.py)

Node types live in nodes.py — subclass a base there to add one. Graph
evaluation (the typed multi-port dataflow, signatures, cross-run cache) lives
in graph.py; this module is the HTTP layer on top of it.
"""

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import chat
from graph import RUNS, Graph, NodeError, _ancestors, _incoming, _sigs, clear_cache, evaluate
from nodes import MEDIA, PROGRESS, REGISTRY, VIDEO_EXTS, Cancelled, free_models


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # the chat model's 16k-ctx variant (see chat.ensure_model). Non-fatal:
    # Ollama down must not take the rest of the backend with it — /chat
    # retries the creation per request and 502s with the real reason.
    try:
        chat.ensure_model()
    except Exception as e:
        print(f"chat: could not prepare Ollama model (will retry on first /chat): {e}")
    yield


app = FastAPI(title="KumoFlow", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Uploads from the browser's file picker, under the OS tmpdir (cleaned by the
# OS, no eviction of our own). Encoded previews/exports live in nodes.MEDIA.
UPLOADS = Path(tempfile.gettempdir()) / "kumoflow_uploads"

_PLAYBACK_FPS = 8  # fallback when the native rate can't be read back (base64 uploads, broken headers)

PREVIEW_MAX_FRAMES = 2000  # video-flattened preview cap — a 100k-frame clip must not become 100k JPEGs


def _native_fps(data):
    """The frame rate in the header of a load_video source on disk (first file
    of a directory dataset), or None for base64 uploads / unreadable files."""
    import cv2

    first = data[0] if isinstance(data, list) else data
    if not isinstance(first, str) or not first or first.startswith("data:"):
        return None
    p = Path(first).expanduser()
    if p.is_dir():
        p = next((q for q in sorted(p.rglob("*")) if q.suffix.lower() in VIDEO_EXTS), p)
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or None


def _upstream_fps(target, by_id, edges):
    """The fps the load_video feeding `target` sampled at, so previews play in
    real time (a clip sampled at 1fps lasts as long as its source footage).
    fps=0 (the default) decoded at the native rate — read it back from the
    file header; base64 uploads fall back.
    ponytail: when max_frames capped the sampling the effective rate is lower
    and playback runs proportionally fast — thread per-clip rates through if
    that ever matters."""
    seen, stack = {target}, [target]
    while stack:
        nid = stack.pop()
        if by_id[nid].kind in ("load_video", "load_stream"):
            fps = float(by_id[nid].config.get("fps") or 0.0)  # falsy covers the field default and an emptied settings field ("")
            return fps if fps > 0 else _native_fps(by_id[nid].config.get("data")) or _PLAYBACK_FPS
        nxt = [e.source for e in edges if e.target == nid and e.source not in seen]
        seen.update(nxt)
        stack += nxt
    return _PLAYBACK_FPS


def _clip_webm(clip, fps, max_side=512):
    """(T,3,H,W) uint8 clip -> /media URL of a downscaled VP8 .webm preview
    encoded at `fps` (the rate the clip was sampled at = real-time playback).

    A real video file instead of per-frame images: the browser's <video>
    element streams it on demand (FileResponse answers Range requests), so
    nothing but the watched clip ever crosses the wire. Content-addressed so
    re-runs of an unchanged graph reuse the file. cv2 is the encoder for the
    same reason it's the decoder in LoadVideo: the only codec dep we have.

    One streaming pass, ~64 frames at a time: `clip` may be an mmap-backed
    view of a spilled file far bigger than RAM (Workstream C), so this never
    materializes a full contiguous copy. Hashing and encoding ride the same
    pass — the hash is only known once the pass is done, so (unlike before)
    every call re-encodes, even a content-addressed hit: a full pass over the
    clip is unavoidable either way.
    """
    import cv2
    from torchvision.transforms.v2.functional import resize

    h, w = clip.shape[-2:]
    th, tw = (round(h * max_side / max(h, w)), round(w * max_side / max(h, w))) if max(h, w) > max_side else (h, w)
    MEDIA.mkdir(exist_ok=True)
    fh = tempfile.NamedTemporaryFile(dir=MEDIA, suffix=".webm", delete=False)
    tmp = Path(fh.name)
    fh.close()
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"VP80"), fps, (tw, th))
    sha1 = hashlib.sha1()
    try:
        for i in range(0, len(clip), 64):
            piece = clip[i : i + 64]
            if (th, tw) != (h, w):
                piece = resize(piece, [th, tw])
            frames = piece.permute(0, 2, 3, 1).cpu().numpy()  # THWC, RGB
            sha1.update(frames.tobytes())
            for f in frames:
                wr.write(f[:, :, ::-1])  # RGB -> BGR
    finally:
        wr.release()
    sha1.update(str(fps).encode())
    dest = MEDIA / f"{sha1.hexdigest()[:20]}.webm"
    if dest.exists():
        tmp.unlink()
    else:
        tmp.rename(dest)
    return f"/media/{dest.name}"


def _volume_raw(clip):
    """(T,3,H,W) uint8 clip -> {url, dims} of a raw grayscale (T,H,W) volume
    file; View Volume fetches it once and reslices on any axis client-side.
    Content-addressed like _clip_webm.
    ponytail: uncompressed over the wire — gzip when localhost stops being
    the deployment story."""
    vol = clip.float().mean(1).round().byte().cpu().numpy()  # (T,H,W); mean folds true-RGB DICOM to gray
    data = vol.tobytes()
    dest = MEDIA / f"{hashlib.sha1(data).hexdigest()[:20]}.raw"
    if not dest.exists():
        MEDIA.mkdir(exist_ok=True)
        dest.write_bytes(data)
    return {"url": f"/media/{dest.name}", "dims": list(vol.shape)}


def _volume_seg(clip, masks, ids, labels):
    """One clip's per-frame masks -> a (T,H,W) uint8 label volume shipped next
    to _volume_raw's gray one: voxel = instance index + 1 (track id when ids
    are wired, else per-frame order), 0 = background. One name/color per
    instance index rides along so the frontend tints and captions with zero
    palette logic — colors match View Video's (_PALETTE / _track_color).
    ponytail: uint8 caps at 255 instances; overlapping masks flatten last-wins."""
    import numpy as np

    from nodes import _PALETTE, _track_color

    h, w = clip.shape[-2:]
    seg = np.zeros((len(masks), h, w), np.uint8)
    names, colors = [], []
    for t, frame in enumerate(masks):
        for j, m in enumerate(frame):
            i = int(ids[t][j]) if ids is not None else j
            if not 0 <= i < 255:
                continue  # -1 flicker tracks and past-the-cap instances stay background
            seg[t][m.bool().cpu().numpy()] = i + 1
            while len(colors) <= i:
                colors.append(None)
                names.append(None)
            colors[i] = list(_track_color(i) if ids is not None else _PALETTE[i % len(_PALETTE)])
            if names[i] is None and labels is not None and labels[t][j]:
                names[i] = labels[t][j]
    data = seg.tobytes()
    dest = MEDIA / f"{hashlib.sha1(data).hexdigest()[:20]}.raw"
    if not dest.exists():
        MEDIA.mkdir(exist_ok=True)
        dest.write_bytes(data)
    return {"seg": f"/media/{dest.name}", "names": names, "colors": colors}


@app.get("/nodes")
def get_nodes():
    return [cls.spec() for cls in REGISTRY.values()]


@app.get("/agent-setup")
def agent_setup():
    # the MCP registration command needs this checkout's absolute path — only
    # the server knows where it lives (the frontend shows a copyable command)
    return {"backend_dir": str(Path(__file__).resolve().parent)}


# Live canvas sync: the open editor and
# external agents share one server-held session graph — editor-shape JSON,
# stored verbatim, rev-checked last-write-wins. Not persisted across restarts;
# the browser's localStorage remains the durable copy.
# ponytail: one global session — keyed sessions only if this ever goes multi-user
SESSION = {"workflow": None, "rev": 0}


class WorkflowPut(BaseModel):
    workflow: dict
    rev: int


@app.get("/workflow")
def get_workflow():
    return SESSION


@app.put("/workflow")
def put_workflow(req: WorkflowPut):
    # structural check only — the editor legitimately holds half-built graphs;
    # agents are expected to validate before push, the server doesn't enforce it
    if not isinstance(req.workflow.get("nodes"), list) or not isinstance(req.workflow.get("edges"), list):
        raise HTTPException(422, "workflow must be {nodes: [...], edges: [...]}")
    if req.rev != SESSION["rev"]:
        return JSONResponse(SESSION, status_code=409)  # current state in the body — the loser rebases without a second GET
    SESSION["workflow"] = req.workflow
    SESSION["rev"] += 1
    return {"rev": SESSION["rev"]}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatBrief(BaseModel):
    input_kind: Literal["image", "video"] | None = None
    output_handling: Literal["view", "export", "app"] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)  # full history, frontend caps at last 20
    graph: dict = {}  # toBackendPayload() output, may be empty
    brief: ChatBrief = ChatBrief()  # last known, echoed back by the frontend


# sync like /run: agent.run_sync blocks for the model's whole turn, and the
# threadpool keeps the rest of the API answering meanwhile. Stateless — the
# frontend owns history, graph and brief.
@app.post("/chat")
def chat_endpoint(req: ChatRequest) -> chat.ChatTurn:
    free_models()  # the chat LLM needs the VRAM the graph's models hold
    try:
        return chat.chat_turn(
            [m.model_dump() for m in req.messages], req.graph, req.brief.input_kind, req.brief.output_handling
        )
    except Exception as e:  # Ollama unreachable, model missing, no parseable turn
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e


# Browser file picks come here as multipart (base64-in-JSON breaks past ~400MB,
# V8's max string length) and the graph carries the returned server paths —
# uploaded once, not re-sent on every run.
@app.post("/upload")
def upload(files: list[UploadFile]):
    UPLOADS.mkdir(exist_ok=True)
    paths = []
    for f in files:
        # full suffix chain, not just the last: nibabel needs ".nii.gz" to stay intact
        dest = UPLOADS / f"{uuid.uuid4().hex}{''.join(Path(f.filename or '').suffixes)}"
        with dest.open("wb") as w:
            shutil.copyfileobj(f.file, w)
        paths.append(str(dest))
    return {"paths": paths}


# Streams encoded previews to <video> tags; FileResponse answers Range
# requests, so the browser fetches only what it plays or seeks to. The route
# param can't contain "/", which pins access inside MEDIA.
@app.get("/media/{name}")
def media(name: str):
    return FileResponse(MEDIA / name)


def _preview_url(jpeg: bytes):
    """A preview's bytes -> its /media URL. Content-addressed like _clip_webm,
    so re-runs of an unchanged graph reuse the file."""
    dest = MEDIA / f"{hashlib.sha1(jpeg).hexdigest()[:20]}.jpg"
    if not dest.exists():
        MEDIA.mkdir(exist_ok=True)
        dest.write_bytes(jpeg)
    return f"/media/{dest.name}"


def _shape(target, outs, by_id, edges):
    """One target's {port: value} outputs -> the JSON its node renders."""
    from nodes import batch_to_previews

    if "download" in outs:  # Export: the zip is already on disk, hand over its URL
        return outs
    if "classification" in outs:  # non-image output: JSON-serializable {labels, scores} per image
        return {"classification": outs["classification"]}
    if "counts" in outs:  # non-image output: {concepts, per_image, total} table
        return {"counts": outs["counts"]}
    if "points" in outs:  # non-image output: scatter points (+ hover thumbnails when images are wired)
        return {k: outs[k] for k in ("points", "thumbnails") if k in outs}
    if by_id[target].kind == "view_video":  # playback: one streamable webm URL per clip
        fps = _upstream_fps(target, by_id, edges)
        res = {"videos": [_clip_webm(clip, fps) for clip in outs["video"]]}
        if outs.get("legend"):  # detected objects: color-pick chips under the player
            res["legend"] = outs["legend"]
        if outs.get("legend_more"):  # entries past the legend's chip cap
            res["legend_more"] = outs["legend_more"]
        return res
    if by_id[target].kind == "view_volume":  # reslicing happens client-side: raw grayscale volume per clip
        vols = [_volume_raw(clip) for clip in outs["video"]]
        masks, ids, labels = (outs.get(k) for k in ("masks", "ids", "labels"))
        if masks is not None:  # a label volume rides along, resliced the same way
            for ci, (v, clip) in enumerate(zip(vols, outs["video"])):
                v.update(_volume_seg(clip, masks[ci], ids[ci] if ids else None, labels[ci] if labels else None))
        return {"volumes": vols}
    img = outs.get("image")
    frame_indices = None  # set below only for a capped video-flattened preview
    if img is None and "video" in outs:  # video target: every frame, flattened clip-major —
        # the Visual Prompt pages through them to draw on any frame (box-draw
        # nodes still use item 0 = clip 0's first frame). ~1 ms JPEG per frame,
        # content-addressed on disk, and the browser fetches only what's shown.
        img = [f for clip in outs["video"] for f in clip]
        if len(img) > PREVIEW_MAX_FRAMES:  # a 100k-frame clip must not become 100k JPEGs
            import torch

            frame_indices = torch.linspace(0, len(img) - 1, PREVIEW_MAX_FRAMES).round().long().tolist()
            img = [img[i] for i in frame_indices]
    if img is None:
        raise ValueError("target node produces no image to display")
    # /media URLs, not data URLs: a node shows one item at a time, so the
    # browser fetches only the one on screen (and caches it) instead of dragging
    # the whole batch through the response. `sizes` is the source resolution —
    # previews are downscaled, and boxes/points drawn on one are in source
    # pixels, so the frontend can't read the scale off the preview itself.
    # Computed from `img` after the cap above, so it stays aligned with `images`.
    res = {
        "images": [_preview_url(p) for p in batch_to_previews(img)],
        "sizes": [[int(im.shape[-1]), int(im.shape[-2])] for im in img],
    }
    if frame_indices is not None:  # true flattened indices the capped preview stands in for
        res["frame_indices"] = frame_indices
    if outs.get("legend"):  # View Image's detected objects: color-pick chips under the pager
        res["legend"] = outs["legend"]
    if outs.get("legend_more"):  # entries past the legend's chip cap
        res["legend_more"] = outs["legend_more"]
    return res


# sync endpoint on purpose: FastAPI runs it in a threadpool, so a slow graph
# run doesn't block other requests — that's the concurrency story (and what
# lets /progress and /stop answer while a run is grinding away).
@app.post("/run")
def run_graph(graph: Graph):
    # 12 GB card can't hold gpt-oss and SAM3 at once — evict the chat model
    # first (/chat evicts nodes._MODELS symmetrically).
    # ponytail: unconditional — sniffing whether the graph will load a GPU
    # model costs more than the occasional needless chat-model reload.
    try:
        chat.unload_model()
    except Exception:
        pass  # Ollama down or model never created — nothing to evict
    by_id = {n.id: n for n in graph.nodes}
    incoming = _incoming(graph.edges)
    upstream = {}
    for e in graph.edges:
        upstream.setdefault(e.target, []).append(e.source)
    # registered before the sig hash: _sigs is expensive (SHA1 over full
    # configs, cache_extra stats), and /stop must find the run in that window
    run = RUNS[graph.run_id] = {
        "cancel": False,
        "node": "",
        "nodes": {},
        "targets": list(dict.fromkeys(graph.targets)),  # dedup, keep order
        "upstream": upstream,
        "cancelled": set(),
        "live": None,  # None = every node; a per-target /stop replaces it
    }
    token = PROGRESS.set(run)
    cache = {}  # shared: a node feeding two targets still runs once
    results = {}
    try:
        sigs = _sigs(by_id, incoming)  # also rejects (hand-POSTed) cyclic graphs
        run["sigs"] = set(sigs.values()) - {None}  # pin against eviction until RUNS.pop below
        for target in run["targets"]:
            if run["cancel"] or target in run["cancelled"]:  # stopped before its turn came
                results[target] = {"cancelled": True}
                continue
            try:
                outs = evaluate(target, by_id, incoming, cache, run, sigs)
                # stopped mid-evaluation but every remaining node was shared with
                # a live target (or fully cached), so it completed anyway — still
                # cancelled from the user's point of view: don't repaint their node
                results[target] = (
                    {"cancelled": True}
                    if run["cancel"] or target in run["cancelled"]
                    else _shape(target, outs, by_id, graph.edges)
                )
            except Cancelled:
                results[target] = {"cancelled": True}  # not an error: the user pressed Stop
            except Exception as e:  # one bad target doesn't cost the others their result
                results[target] = {"error": str(e)}
                if isinstance(e, NodeError):
                    results[target]["node"] = e.node_id  # which node failed, so the UI stops there
    finally:
        PROGRESS.reset(token)
        RUNS.pop(graph.run_id, None)
    return {"results": results}


# Workflow snapshots for deploy.py — the exact /run payload (uploads' data URLs
# and all) plus an optional "ui" page spec, so the pipeline runs without the
# frontend. POST /deploy also puts the artifact live: it spawns deploy.py as a
# child process and tracks it here. Children die with the backend (atexit);
# artifact files persist, so everything is one click (or one manual command)
# from being live again after a restart.
DEPLOYMENTS = Path(__file__).parent / "deployments"
PROCS: dict[str, dict] = {}  # deployment name -> {"port", "proc", "has_ui"}


class DeployRequest(BaseModel):
    graph: Graph
    ui: dict | None = None  # the App node's page spec; absent on API-only deploys
    name: str = "pipeline"
    port: int = 8001


def _terminate(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@atexit.register
def _stop_children():
    for d in PROCS.values():
        _terminate(d["proc"])


@app.post("/deploy")
def deploy(req: DeployRequest):
    DEPLOYMENTS.mkdir(exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", req.name).strip("._") or "pipeline"
    # deployment identity = name: redeploying a name replaces it in place (same
    # port by default, URL stays stable); another deployment's port is a conflict
    for other, d in PROCS.items():
        if other != name and d["port"] == req.port and d["proc"].poll() is None:
            raise HTTPException(409, f"port {req.port} is already used by deployment {other!r}")
    dest = DEPLOYMENTS / f"{name}.json"
    artifact = req.graph.model_dump(exclude={"run_id"})
    if req.ui is not None:
        artifact["ui"] = req.ui
    dest.write_text(json.dumps(artifact, indent=1))
    old = PROCS.get(name)
    if old:
        _terminate(old["proc"])
    proc = subprocess.Popen(
        ["uv", "run", "deploy.py", f"deployments/{dest.name}"],
        cwd=Path(__file__).parent,
        env={**os.environ, "PORT": str(req.port)},
    )
    PROCS[name] = {"port": req.port, "proc": proc, "has_ui": req.ui is not None}
    return {
        "name": name,
        "port": req.port,
        "url": f"http://localhost:{req.port}",
        "path": str(dest),
        "command": f"PORT={req.port} uv run deploy.py deployments/{dest.name}",
    }


# `alive` via poll(), so a crashed or OOM-killed deployment shows honestly.
@app.get("/deployments")
def deployments():
    return [
        {"name": name, "port": d["port"], "has_ui": d["has_ui"], "alive": d["proc"].poll() is None}
        for name, d in PROCS.items()
    ]


@app.delete("/deployments/{name}")
def stop_deployment(name: str):
    d = PROCS.pop(name, None)
    if d is None:
        raise HTTPException(404, f"no deployment named {name!r}")
    _terminate(d["proc"])  # the artifact file stays — restartable any time
    return {"stopped": name}


# The sig can't see state outside the graph (files a Custom Code node reads, a
# network resource): after changing such state, this is the user's escape hatch.
@app.post("/cache/clear")
def cache_clear():
    return {"cleared": clear_cache()}


# Polled while a run is in flight: {node id: [work units done, total]} for the
# nodes that report (models, tracking, video decode). Empty once the run ends.
@app.get("/progress/{run_id}")
def progress(run_id: str):
    run = RUNS.get(run_id)
    # snapshot: the run's own thread adds a key here whenever a node declares its
    # work, and serializing the live dict would blow up mid-iteration
    return {"nodes": dict(run["nodes"]) if run else {}}


# Stop a run: the flag is picked up at the next node boundary or work unit, so
# the /run request still returns — with {"cancelled": true} per target.
# With ?target=<node id>, stop just that target's subgraph: the live set is
# recomputed as everything the remaining targets still need, so nodes shared
# with an uncancelled target keep running and their results stay cached.
@app.post("/stop/{run_id}")
def stop(run_id: str, target: str | None = None):
    run = RUNS.get(run_id)
    if run:
        if target is None:
            run["cancel"] = True
        else:
            run["cancelled"].add(target)
            live = set()
            for t in run["targets"]:
                if t not in run["cancelled"]:
                    live |= _ancestors(t, run["upstream"])
            run["live"] = live  # one atomic swap — the run thread reads it lock-free
            for nid in list(run["nodes"]):  # drop dead nodes' bars from /progress polls
                if nid not in live:
                    run["nodes"].pop(nid, None)
    return {"stopping": run is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get("PORT", "8000")))
