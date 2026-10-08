# How a graph run works

`POST /run` evaluates the flow graph up to its `targets` — one node (the ▶ Run
you clicked) or every display node on the canvas (▶ Run All), all in a single
pass over a shared cache. The engine is `evaluate()` in `backend/graph.py`: a
pull-based, memoized recursion split into two closures that mirror the dataflow
itself —

- `node_outputs(nid)` — run one node (inputs first) → its `{port: value}` dict, memoized
- `port_value(node, port)` — the value one connected input port receives from its edge(s)

They call each other until they bottom out at source nodes with no inputs.
There is no explicit topological sort: the depth-first recursion **is** the
execution order, and a per-request `cache` ensures every node runs at most once.

## Worked example: SAM3 text-prompted bounding boxes

Four nodes, wired as you'd click them together in the UI. Note the diamond:
`load-1` feeds **both** SAM3 and the viewer — that's what makes the cache
interesting.

```
        load-1 (Load Image, photo of a street)
        /            \
  image│              │image
       ▼              │
   sam3-3 ◄──prompts──┤── text_prompt-2 (text="car")
       │              │
  boxes│              │
       ▼              ▼
      view-4 (View Image)      (masks port: left unwired)
```

### Step 0 — what the backend receives

The frontend POSTs the whole graph with `targets: ["view-4"]` and a `run_id` it
made up (that's what `/progress` and `/stop` are keyed on). `run_graph` builds
two lookup structures and calls `evaluate` once per target:

- `by_id` = `{"load-1": NodeIn(...), "text_prompt-2": NodeIn(config={"text": "car"}), ...}`
- `_incoming(edges)` flips the edges into "what feeds this input port":

```python
{
  ("sam3-3", "image"):   [("load-1", "image")],
  ("sam3-3", "prompts"): [("text_prompt-2", "prompts")],
  ("view-4", "image"):   [("load-1", "image")],
  ("view-4", "boxes"):   [("sam3-3", "boxes")],
}
```

### Step 1 — `evaluate("view-4", by_id, incoming)`

Creates `cache = {}`, defines the two closures over `by_id`/`incoming`/`cache`
(which is why the recursive calls need no parameter threading), and kicks
everything off with `return node_outputs("view-4")`.

### Step 2 — `node_outputs("view-4")`

Not cached. Looks up `cls = ViewImage` in `REGISTRY`, instantiates it (pydantic
validates the config here). The kwargs comprehension walks
`cls.inputs = ["image", "boxes", "masks"]` asking one question per port:
*is `("view-4", port)` a key in `incoming`?*

- `image` → yes → `port_value(view, "image")`
- `boxes` → yes → `port_value(view, "boxes")`
- `masks` → **no** → no kwarg at all; `run(image, boxes=None, masks=None)`'s
  own default fills it. Unwired optional ports cost nothing.

### Step 3 — `port_value(view, "image")` — the left side of the diamond

One edge feeds it: `("load-1", "image")`, so it needs `node_outputs("load-1")`:

- Not cached → `LoadImage(data="data:image/png;base64,...")` → `run()` decodes
  the upload into a batch: a length-1 list `[Tensor(3,H,W)]`.
- `run()` returned a bare list, not a dict → the normalization line wraps it as
  `{"image": [tensor]}` using the first declared output port.
- `cache["load-1"] = {"image": [...]}` ← first cache write.

Back in `port_value`: exactly one value → pass-through, no merging.

### Step 4 — `port_value(view, "boxes")` — where the real work is

One edge: `("sam3-3", "boxes")` → `node_outputs("sam3-3")`:

- `Sam3(threshold=0.3)` instantiated, then *its* kwargs comprehension recurses
  another level down over `["image", "prompts"]`:
  - `port_value(sam3, "image")` → `node_outputs("load-1")` → **cache hit** —
    the very same batch object, instantly. The image is decoded once per run,
    no matter how many nodes consume it.
  - `port_value(sam3, "prompts")` → `node_outputs("text_prompt-2")` →
    `TextPrompt(text="car").run()` → already a dict,
    `{"prompts": {"text": "car"}}`, cached. One edge → passes through unmerged.
    (Wire a second Text Prompt `"truck"` into the same port and `len(vals) == 2`
    triggers `_merge_prompts`, handing SAM3 `{"text": "car, truck"}` — texts
    comma-join, point/box lists concatenate.)
- Required-port check: `image` ✓ `prompts` ✓. An unwired prompt fails here with
  `"input 'prompts' not connected"` instead of a `TypeError` from `run()`.
- `inst.run(image=batch, prompts={"text": "car"})` — no click points in the
  prompt dict, so SAM3 takes the concept-model path: text broadcast across the
  batch, every matching instance returned →
  `{"boxes": [per-image box tensors], "masks": [...]}`. Already a dict → cached
  as-is.

`port_value` plucks the `"boxes"` entry out of that dict; the `masks` output is
computed but never consumed — nobody pulls on it.

### Step 5 — back in `node_outputs("view-4")`, unwound

`kwargs = {"image": batch, "boxes": [...]}`. Required check passes (`image` is
ViewImage's only required port). `run()` draws a colored rectangle per detection
on a copy of each frame, returns the drawn batch → wrapped as `{"image": ...}`
→ cached → returned up through `evaluate`.

### Step 6 — back in `/run`

`outs` has `"image"`, so `_shape()` encodes each frame as a downscaled JPEG under
`/media` and sends the URLs plus each image's source size, so the frontend's pager
fetches only the item on screen. (A classification target is shaped as
`{"classification": ...}` JSON instead.) The response is keyed by target —
`{"results": {"view-4": {"images": [...]}}}` — and each target is shaped inside
its own `try`, so one failing viewer doesn't cost the others their result.

## Progress and stopping

A node that takes real time (models, tracking, video decode) declares its work
units with `_total(n)` and ticks them off with `_step()` (`backend/nodes.py`);
everything routed through `_chunked()` ticks per forward pass for free. Those
counts land in `RUNS[run_id]["nodes"]`, which `GET /progress/{run_id}` serves
and the frontend polls every 400ms to fill each node's bar.

`POST /stop/{run_id}` sets a flag in the same slot. `_step()` raises `Cancelled`
when it sees it, and `evaluate` checks it once per node before running it — so a
stop lands either between nodes or between work units, and the `/run` request
still returns, with `{"cancelled": true}` for the targets that didn't finish.
Whatever nodes already *completed* stay in the result cache (below) — only the
node a stop interrupted is discarded; the UI keeps the previous results on
screen. A target whose whole subgraph is served from the result cache runs no
node at all — after a stop it still answers `{"cancelled": true}` instead of
repainting.

## The result cache

Results persist across runs: a process-global LRU (`CACHE` in
`backend/graph.py`) means editing only the tail of a graph re-runs only the
tail — the expensive upstream (a SAM3 sweep over a whole dataset) is served
instantly. There is no run-to-run diffing; before evaluation, `_sigs` computes
a structural (Merkle) signature per node from the graph JSON alone (a full
walkthrough with a worked example lives in
[structural-signatures.md](structural-signatures.md)):

    sig(node) = sha1(kind, canonical_json(validated config), cache_extra(),
                     [(input_port, src_output_port, sig(src)) per incoming edge])

Any change to a node's kind/config/wiring — or anything upstream — changes its
sig and, transitively, all its descendants'; everything else keeps its sig and
hits. Node ids are excluded, so a copy-pasted identical branch hits too. In
`node_outputs` the global cache is consulted *before* recursing into inputs,
so a hit skips the entire upstream subgraph, not just one node. Failures and
cancellations are never cached (the exception propagates before the write).

Two hooks on the `Node` base class — a new node needs neither:

- `cacheable: ClassVar[bool] = True` — set `False` on a nondeterministic node,
  or on one whose result isn't worth an entry (`filter_class` re-runs in
  milliseconds and its output aliases its input batch): its sig becomes `None`
  (always runs) and poisons all descendants to `None`.
- `cache_extra(self) -> str` — extra key material for state outside
  config+inputs. `LoadVideo` returns `(path, mtime_ns, size)` per file in
  filesystem-path mode, so editing a video on disk invalidates the cache.

The byte budget is `FLOW_CACHE_BYTES` (default 4 GiB), least-recently-used
evicted on insert — but entries an in-flight run still needs (its sig set,
registered by `/run` as `run["sigs"]`) are skipped: evicting one before the
run first touches it would force a full recompute, so the cache may sit over
cap until those runs finish. Determinism is assumed, not enforced: an impure
Custom Code node (random/network/file reads) will serve stale results while
its code and inputs are unchanged — `POST /cache/clear` is the escape hatch.

## The shape of it

- **Execution order = recursion order.** `view-4` asked for its image (load
  ran), then its boxes (text prompt ran, then SAM3 ran). Nobody sorted anything.
- **The cache turns the diamond back into "each node runs exactly once":**
  4 `node_outputs` executions, 1 cache hit, and `text_prompt-2` never ran until
  something actually pulled on it.
- **The per-request cache spans its targets.** That is what makes ▶ Run All
  cheap: `View Image ← SAM3 → View Image with mask` are two targets over one
  cache, so SAM3 runs once. Across requests, the global result cache (above)
  adds the incrementality: a re-run only executes what changed.
- **Cycles are rejected up front — among cacheable nodes.** The frontend's
  `createsCycle` guard refuses to create them; a hand-crafted cyclic payload
  trips the visiting-set in `_sigs` before anything runs — unless the cycle
  routes through an uncacheable node (`_sigs` returns early without visiting
  its inputs), in which case it dies later as that target's RecursionError.
