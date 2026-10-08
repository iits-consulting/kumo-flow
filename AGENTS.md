# KumoFlow — Agent Guide

KumoFlow is a node-based computer-vision workflow editor: a Svelte canvas
(frontend) wired to a FastAPI + torch backend that runs graphs of nodes
(load, segment, track, classify, view, ...). Workflows are plain JSON files;
you author and edit them directly. Start the backend with `make up` from the
repo root — it serves on `http://localhost:8000` (`make up PORT=8001` to
override). Prefer keeping the server running: `cli.py run` defaults to it
(warm models, shared cache) and only falls back to slow in-process execution.

**The loop:** author the workflow JSON → validate → fix errors → run → read
results.

Before declaring a workflow done: map every clause of the user's request to a
node or config in the graph; a clause the catalog can't express must be
reported to the user, never approximated silently. After a run, fetch the
result media (`--out` / `fetch_media`) and look at it — confirm each clause is
visibly satisfied.

## CLI

All commands run from `backend/`:

```
uv run cli.py catalog [--input-kind image|video]     # machine-readable node list (JSON)
uv run cli.py validate <workflow.json> [--target ID ...]
uv run cli.py run <workflow.json> [--target ID ...] [--out DIR] [--local] [--run-id ID]
uv run cli.py pull [file]                            # live editor canvas -> file (default stdout)
uv run cli.py push <workflow.json>                   # replace the live canvas (validate first!)
```

Examples:

```
uv run cli.py catalog --input-kind image
uv run cli.py validate ../my-workflow.json
uv run cli.py run ../my-workflow.json --out /tmp/results
```

`validate` prints one finding per line prefixed `error:` / `warning:`.
Exit codes: `0` valid, `1` errors, `2` file/parse failure. `run` prints the
results JSON to stdout; `--out DIR` downloads produced media there and
rewrites the JSON to local paths so you can `Read` the images. Targets
default to every Output-category node in the file; pass `--target ID` to
override.

## Workflow file schema

A workflow is `{nodes: [...], edges: [...]}`. Minimal real example
(load → blur → view) — this exact file validates:

```json
{
  "nodes": [
    {
      "id": "load-1",
      "type": "flow",
      "position": { "x": 0, "y": 0 },
      "data": {
        "label": "Load Image",
        "kind": "load",
        "color": "#16a34a",
        "inputs": [],
        "outputs": ["image", "labels"],
        "required_inputs": [],
        "config": { "data": "", "labels": [] }
      }
    },
    {
      "id": "blur-2",
      "type": "flow",
      "position": { "x": 250, "y": 0 },
      "data": {
        "label": "Blur",
        "kind": "blur",
        "color": "#2563eb",
        "inputs": ["image", "video", "masks"],
        "outputs": ["image", "video"],
        "required_inputs": [],
        "config": { "blur": 25 }
      }
    },
    {
      "id": "view-3",
      "type": "flow",
      "position": { "x": 500, "y": 0 },
      "data": {
        "label": "View Image",
        "kind": "view",
        "color": "#9333ea",
        "inputs": ["image", "boxes", "masks", "labels"],
        "outputs": [],
        "required_inputs": ["image"],
        "config": { "colors": {} }
      }
    }
  ],
  "edges": [
    { "id": "e1", "source": "load-1", "target": "blur-2", "sourceHandle": "image", "targetHandle": "image" },
    { "id": "e2", "source": "blur-2", "target": "view-3", "sourceHandle": "image", "targetHandle": "image" }
  ]
}
```

Annotations:

- Node `id` convention: `{kind}-{n}` with `n` unique across the file.
- `position` is required but cosmetic — it never affects execution. Layout
  guidance: left-to-right by depth, ~250 px per column, ~150 px per row.
- `data.kind` and `data.config` are what the backend reads. Config keys and
  defaults come from `catalog` output.
- `label`, `color`, `inputs`, `outputs`, `required_inputs` in `data` are
  display copies for the editor — copy them verbatim from the node's
  `catalog` entry when authoring a node.
- Edges: `sourceHandle`/`targetHandle` name the ports; `null`/missing
  handles are accepted and default to `image`. Name them explicitly.
- The backend/deployment shape (nodes as flat `{id, kind, config}`) is also
  accepted by `validate`/`run`.

## Port contract

Ten port types: `image, video, masks, boxes, prompts, labels,
classification, embedding, ids, overlay`.

- Ports mate when names are equal, plus the one widening: `image → overlay`.
- One edge per input port — except `prompts`, which merges multiple edges.
- Every port in a node's `required_inputs` must be wired for the node to run.

Transforms like blur act on the WHOLE image unless masks are wired into
them. When the goal targets only part of the image (an object, the
background, faces), first get masks via `text_prompt → segment`; when the
target is everything EXCEPT the named subject (e.g. "the background"),
segment the subject and pass its masks through `mask_ops` with
`invert: true`.

## Recipes

Task-shaped compositions (kinds from the catalog; each `→` is an edge):

- Subject on a plain canvas / replace the background: `segment` →
  `mask_ops(invert=true)` → `warp_overlay` — its `color` fills the inverted
  region; wire an image into `overlay` instead for a picture background.
- Blur (or otherwise transform) only the background: segment the subject,
  invert via `mask_ops`, wire the masks into `blur`; skip the invert to
  transform the subject itself.
- Put a logo/image onto a detected object (license plate, billboard): a
  second `load` carries the logo into `warp_overlay`'s `overlay` port;
  `segment`'s masks say where.
- Identify then classify: `segment` → `crop` → `classify` (crop makes each
  detected instance its own batch item).
- Count in a region: `segment` → `filter_region` → `count`.
- Labeled dataset from detections: `crop` carries `labels` — one label per
  crop, aligned with SAM3's per-instance concept names.

## Hard rules

- **Never author or modify `custom` nodes** — they run unsandboxed `exec()`.
  `validate` warns if one is present; leave user-made ones alone.
- Don't embed `data:` URLs in configs — the editor scrubs them on save.
  Leave `load`'s `data` empty (or pass media another way) and let the user
  fill it in the editor.
- Always fetch the catalog (`uv run cli.py catalog`, or the `catalog` MCP
  tool) instead of trusting any hardcoded node list — this file deliberately
  contains no copy of the catalog; `catalog` is the only authoritative source
  of kinds, ports, and config fields.

## MCP server

For persistent wiring instead of shelling out:

```
claude mcp add kumoflow -- uv run --directory <repo>/backend --extra agent mcp_server.py
```

Eight tools, all delegating to the running backend: `catalog`,
`validate_workflow`, `run_workflow`, `get_progress`, `stop_run`,
`fetch_media` (saves media to a local file you can `Read`), and
`get_canvas`/`set_canvas` (read/replace the user's live editor canvas —
the CLI's `pull`/`push`). Requires the backend up (`make up`).
Both `push` and `set_canvas` accept either workflow shape: backend-shape
nodes (`{id, kind, config}`) are auto-converted to editor shape with auto-layout.
