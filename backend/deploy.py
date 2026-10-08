"""Serve one saved workflow as a standalone local API — no frontend needed.

Save a workflow with the editor's Deploy button (which also spawns this, see
main.py's POST /deploy) or hand-write a /run-shaped graph JSON, then:

    uv run deploy.py deployments/<name>.json      # PORT=8001 by default

  GET  /             the app page (frontend/dist/app.html), only when the
                     artifact carries a "ui" spec (an App-node deploy)
  GET  /info         what the pipeline expects: its input nodes, targets,
                     node kinds, and the ui spec (null for API-only deploys)
  POST /run          execute the pipeline; same per-target JSON as the editor.
                     Optional multipart uploads swap input nodes' data for this
                     run: field name = input node id (with a single input node,
                     any field name lands on it). No files = re-run on the
                     deployed inputs.
  GET  /media/{name} images / export zips the results point at

From Python, client.py wraps this:  Pipeline("localhost:8001").run(images={"image": "cat.jpg"})

Runs share main.py's cross-run cache, so an unchanged re-run is instant.
"""

import base64
import json
import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile  # what request.form() actually yields

import main
from main import Graph

INPUT_KINDS = ("load", "load_video", "load_volume")
DIST = Path(__file__).parent.parent / "frontend" / "dist"


def create_app(workflow: Path) -> FastAPI:
    spec = json.loads(workflow.read_text())
    ui = spec.pop("ui", None)  # page spec — not part of the Graph model
    app = FastAPI(title=f"KumoFlow pipeline: {workflow.stem}")
    app.get("/media/{name}")(main.media)

    def graph():  # fresh models per run — a run's input override must not leak into the next
        return Graph(**spec)

    @app.get("/info")
    def info():
        g = graph()
        return {
            "workflow": workflow.stem,
            "inputs": [{"id": n.id, "kind": n.kind} for n in g.nodes if n.kind in INPUT_KINDS],
            "targets": g.targets,
            # the app page resolves each ui section's renderer from these
            "kinds": {n.id: n.kind for n in g.nodes},
            "ui": ui,
        }

    @app.post("/run")
    async def run(request: Request):
        g = graph()
        inputs = {n.id: n for n in g.nodes if n.kind in INPUT_KINDS}
        uploads: dict[str, list[UploadFile]] = {}  # input node id -> its files (several = a batch)
        for key, f in (await request.form()).multi_items():
            if not isinstance(f, UploadFile):
                raise HTTPException(400, f"form field {key!r} must be a file upload")
            nid = key if key in inputs else next(iter(inputs)) if len(inputs) == 1 else None
            if nid is None:
                raise HTTPException(400, f"unknown input {key!r} — this workflow's inputs: {', '.join(inputs) or 'none'}")
            uploads.setdefault(nid, []).append(f)
        for nid, fs in uploads.items():
            # ponytail: uploads ride through as base64 data URLs (the one format
            # both load nodes decode) — spill to disk paths if huge videos hurt
            urls = [
                f"data:{f.content_type or 'application/octet-stream'};base64,"
                + base64.b64encode(await f.read()).decode()
                for f in fs
            ]
            inputs[nid].config = {**inputs[nid].config, "data": urls if len(urls) > 1 else urls[0], "labels": []}
        g.run_id = uuid.uuid4().hex
        # threadpool like the editor's sync /run: a grinding model must not block /media
        return await run_in_threadpool(main.run_graph, g)

    if ui:
        if not (DIST / "app.html").exists():
            raise RuntimeError(f"this deployment has an app page but {DIST / 'app.html'} is missing — run `npm run build` in frontend/")
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/")
        def page():
            return FileResponse(DIST / "app.html")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon():
            return FileResponse(DIST / "favicon.ico")

    return app


if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) != 2:
        sys.exit("usage: uv run deploy.py <workflow.json>")
    uvicorn.run(create_app(Path(sys.argv[1])), host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
