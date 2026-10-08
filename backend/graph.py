"""KumoFlow graph engine — evaluate a flow graph up to a target node.

The graph is a typed multi-port dataflow: edges carry whatever a node's output
port produces (image, prompts, boxes, masks) and are routed by
(sourceHandle -> targetHandle). Node types live in nodes.py. No HTTP in here —
main.py's endpoints are one consumer, deploy.py's pipelines another.
"""

import hashlib
import json
import os
import threading
from collections import OrderedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodes import REGISTRY, Cancelled

# Node kinds that predate the model-dropdown merge (saved graphs, deploys, old
# chat transcripts): old kind -> (new kind, config defaults to inject). Defaults
# are injected only where the graph doesn't set the key itself — needed where
# the merged node's default differs from the old node's (hf_classify was ViT,
# Classify defaults to SigLIP 2). Applied in NodeIn validation, the one choke
# point every entry (main /run, deploy.py, chat-built graphs) parses through.
KIND_ALIASES = {
    "sam3": ("segment", {"model_id": "facebook/sam3"}),
    "hf_segment": ("segment", {"model_id": "facebook/mask2former-swin-tiny-coco-instance"}),
    "siglip2": ("classify", {"model_id": "google/siglip2-base-patch16-224"}),
    "hf_classify": ("classify", {"model_id": "google/vit-base-patch16-224"}),
    "dinov2": ("embed", {"model_id": "facebook/dinov2-base"}),
    "hf_embed": ("embed", {"model_id": "facebook/dinov2-base"}),
    "hf_detect": ("detect", {}),
    "hf_depth": ("depth", {}),
    "hf_caption": ("caption", {}),
    # custom_model merged into custom: input_ports restores the old fixed
    # handles so wired edges survive; old code must rename predict() to
    # run(loaded, ...) — the run-time error says so
    "hf_custom": ("custom", {"input_ports": ["image", "prompts"]}),
    "custom_model": ("custom", {"input_ports": ["image", "prompts"]}),
}


class NodeIn(BaseModel):
    id: str
    kind: str
    config: dict = {}

    @model_validator(mode="after")
    def _resolve_kind_alias(self):
        if alias := KIND_ALIASES.get(self.kind):
            self.kind, defaults = alias
            self.config = {**defaults, **self.config}  # the graph's own config wins
        return self


class Edge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str
    target: str
    # port names; default to "image" so a plain single-image chain needs no handles
    source_handle: str | None = Field("image", alias="sourceHandle")
    target_handle: str | None = Field("image", alias="targetHandle")


class Graph(BaseModel):
    nodes: list[NodeIn]
    edges: list[Edge]
    # every node whose result the frontend wants, evaluated in one pass over a
    # shared cache — "Run All" with two viewers on one SAM3 runs SAM3 once
    targets: list[str]
    # ties this run to GET /progress and POST /stop. ponytail: two concurrent
    # runs that both omit it share the "" slot and only one stays stoppable —
    # the frontend always sends one, so this only bites hand-rolled callers
    run_id: str = ""


# In-flight runs, keyed by the id the frontend makes up per run: {"cancel":
# bool, "node": the node currently running, "nodes": {id: [done, total]},
# "targets"/"upstream": the run's graph shape, "cancelled": targets stopped
# individually, "live": node ids still needed by an uncancelled target (None
# until a per-target stop happens), "sigs": the run's cache signatures, pinned
# against eviction in _cache_put}. Nodes write into it through
# nodes.PROGRESS; /progress reads it, /stop flips the flag or shrinks the live
# set. Entries are created and popped by run_graph.
RUNS: dict[str, dict] = {}

# Cross-run result cache: structural signature (see _sigs) -> the node's
# completed {port: value} outputs. Any later run whose node hashes to the same
# sig reuses the result without touching the node or anything upstream of it.
# LRU, evicted down to FLOW_CACHE_BYTES on insert — but entries an in-flight
# run still needs (its run["sigs"]) are skipped, so eviction never forces a
# live run to recompute something it cached seconds ago; if only pinned
# entries remain, the cache stays over cap until those runs finish. Values are
# shared, never copied: nodes must not mutate inputs or results (see Node
# docstring).
CACHE: OrderedDict[str, dict] = OrderedDict()  # sig -> {"result", "nbytes", "disk"}
CACHE_BYTES = int(os.environ.get("FLOW_CACHE_BYTES", 4 * 2**30))
CACHE_DISK_BYTES = int(os.environ.get("FLOW_CACHE_DISK_BYTES", 64 * 2**30))
_CACHE_LOCK = threading.Lock()  # /run handlers share a threadpool


def _nbytes(v):
    """Approximate (ram, disk) bytes held by a cached value — over-counting is
    the safe direction. A spill-backed tensor (mmap view of an unlinked
    SPILL_DIR file, see nodes._SpillWriter) barely costs RAM at all — its
    storage lives on disk instead, so it's counted there."""
    if hasattr(v, "untyped_storage"):  # torch tensor: a view pins its whole storage
        if v.untyped_storage().filename is not None:  # mmap-backed spill file
            return 64, int(v.untyped_storage().nbytes())
        # ponytail: N views of one storage count N× its size — dedupe by
        # storage data_ptr only if the over-count ever hurts
        return max(int(v.nbytes), int(v.untyped_storage().nbytes())), 0
    if hasattr(v, "nbytes"):  # ndarray
        return int(v.nbytes), 0
    if isinstance(v, dict):
        pairs = [_nbytes(x) for kv in v.items() for x in kv]
        return sum(r for r, _ in pairs), sum(d for _, d in pairs)
    if isinstance(v, (list, tuple)):
        pairs = [_nbytes(x) for x in v]
        return sum(r for r, _ in pairs), sum(d for _, d in pairs)
    if isinstance(v, (str, bytes)):
        return len(v), 0
    return 32, 0  # ints, floats, None, ...


def _cache_get(sig):
    """The cached {port: value} for `sig` (bumped to most-recent), or None."""
    with _CACHE_LOCK:
        entry = CACHE.get(sig)
        if entry is None:
            return None
        CACHE.move_to_end(sig)
        return entry["result"]


def _cache_put(sig, result):
    with _CACHE_LOCK:
        ram, disk = _nbytes(result)
        CACHE[sig] = {"result": result, "nbytes": ram, "disk": disk}
        CACHE.move_to_end(sig)  # assignment to an existing key keeps its old slot
        total = sum(e["nbytes"] for e in CACHE.values())
        total_disk = sum(e["disk"] for e in CACHE.values())
        if total <= CACHE_BYTES and total_disk <= CACHE_DISK_BYTES:
            return
        # sigs some in-flight run still needs: evicting one before that run
        # first touches it forces a full recompute (decode + GPU minutes) to
        # save RAM/disk we have — skip them, over-cap beats recompute
        pinned = {s for r in list(RUNS.values()) for s in r.get("sigs", ())}
        for s in list(CACHE):
            if total <= CACHE_BYTES and total_disk <= CACHE_DISK_BYTES:
                break
            if s not in pinned:
                # dropping the popped entry's last tensor reference here is what
                # actually frees a spill-backed value's unlinked disk file
                e = CACHE.pop(s)
                total -= e["nbytes"]
                total_disk -= e["disk"]


def clear_cache():
    """Drop every cross-run entry; returns how many there were."""
    with _CACHE_LOCK:
        n = len(CACHE)
        CACHE.clear()
    return n


def _sigs(by_id, incoming):
    """{node id: sig hex | None} for every node — a structural (Merkle) hash of
    everything the node's result can depend on: kind, validated config,
    cache_extra(), and the sig of every wired-in output, in the order
    evaluate() consumes the edges (a reordered multi-edge port is a cache miss,
    never a wrong hit). Node ids are excluded, so identical copy-pasted
    branches share entries. None = uncacheable (cacheable=False), and any
    upstream None poisons all descendants to None."""
    sigs: dict[str, str | None] = {}
    visiting = set()

    def sig(nid):
        if nid in sigs:
            return sigs[nid]
        if nid in visiting:
            raise ValueError("graph has a cycle")
        visiting.add(nid)
        sigs[nid] = compute(nid)
        visiting.discard(nid)
        return sigs[nid]

    def compute(nid):
        node = by_id[nid]
        cls = REGISTRY.get(node.kind)
        if cls is None or not cls.cacheable:
            return None
        try:
            inst = cls(**node.config)
            # the validated dump makes explicit-default and omitted hash identically
            cfg, extra = json.dumps(inst.model_dump(), sort_keys=True, default=str), inst.cache_extra()
        except Exception:  # invalid config — run() will raise the real error
            cfg, extra = json.dumps(node.config, sort_keys=True, default=str), ""
        h = hashlib.sha1(f"{node.kind}\0{cfg}\0{extra}".encode())
        for (n, port), srcs in incoming.items():
            if n != nid:
                continue
            for src_id, src_port in srcs:
                s = sig(src_id)
                if s is None:
                    return None
                h.update(f"\0{port}\0{src_port}\0{s}".encode())
        return h.hexdigest()

    for nid in by_id:
        sig(nid)
    return sigs


def _incoming(edges):
    """(target_id, target_port) -> [(source_id, source_port), ...]."""
    m = {}
    for e in edges:
        key = (e.target, e.target_handle or "image")
        m.setdefault(key, []).append((e.source, e.source_handle or "image"))
    return m


def _ancestors(target, upstream):
    """`target` plus every node id feeding it, transitively — the subgraph a
    run computes for that target (mirror of the frontend's ancestors())."""
    seen, stack = {target}, [target]
    while stack:
        for src in upstream.get(stack.pop(), ()):
            if src not in seen:
                seen.add(src)
                stack.append(src)
    return seen


def _merge_prompts(vals):
    """Several prompt edges into one port (e.g. two Text Prompt nodes into SigLIP):
    text fields join comma-separated, point/box lists concatenate. Note: two
    Visual Prompts therefore append their `per_image` batches instead of merging
    frame-wise (SAM3 truncates to its batch size) — wire only one per port."""
    out = {}
    for v in vals:
        for k, x in v.items():
            prev = out.get(k)
            if isinstance(x, str):
                out[k] = ", ".join(s for s in (prev, x) if s)
            elif isinstance(x, list) and isinstance(prev, list):
                out[k] = prev + x
            else:
                out[k] = x
    return out


class NodeError(ValueError):
    """A node failed to validate or run — carries the node id so the frontend
    can pin the error on that node instead of the run's target."""

    def __init__(self, node_id: str, msg: str):
        super().__init__(msg)
        self.node_id = node_id


def evaluate(target, by_id, incoming, cache=None, run=None, sigs=None):
    """Evaluate the graph upstream of `target`; return its {port: value} outputs.

    Pull-based recursion in two halves that mirror the dataflow: a node's outputs
    are produced by running it on its ports' values, and a port's value is the
    output of the node(s) wired into it — so `node_outputs` and `port_value`
    call each other until they bottom out at source nodes (no inputs).

    `cache` is shared across the targets of one request; `run` is that request's
    progress/cancel slot (see RUNS) — pass None to evaluate untracked. `sigs`
    (from _sigs) keys completed results into the cross-run CACHE — pass None to
    skip it."""
    cache = {} if cache is None else cache
    sigs = sigs or {}

    def port_value(node, port):
        """The value one connected input port receives: a single edge passes its
        source output through; several edges merge for `prompts`, error elsewhere."""
        vals = []
        for src_id, src_port in incoming[(node.id, port)]:
            src_out = node_outputs(src_id)
            if src_port not in src_out:
                raise ValueError(f"{node.kind!r} input {port!r}: upstream has no output {src_port!r}")
            vals.append(src_out[src_port])
        if len(vals) == 1:
            return vals[0]
        if port == "prompts":
            return _merge_prompts(vals)
        raise ValueError(f"{node.kind!r} input {port!r}: only one connection allowed")

    def node_outputs(nid):
        """Run one node (evaluating its inputs first) -> its {port: value} dict, memoized."""
        if nid in cache:
            return cache[nid]
        sig = sigs.get(nid)
        if sig is not None:
            # checked before recursing into inputs: a hit skips the whole
            # upstream subgraph, which is the point of the cross-run cache
            hit = _cache_get(sig)
            if hit is not None:
                cache[nid] = hit
                return hit
        node = by_id[nid]
        cls = REGISTRY.get(node.kind)
        if cls is None:
            raise ValueError(f"unknown node kind: {node.kind!r}")
        try:
            inst = cls(**node.config)  # pydantic validates the config here

            # only connected ports get a kwarg; run()'s defaults cover optional
            # ones. Keyed off the actually-wired ports (not cls.inputs) so nodes
            # with dynamic ports (CustomNode) work the same as fixed ones.
            kwargs = {port: port_value(node, port) for (n, port) in incoming if n == nid}
            for port in cls.required_ports():
                if port not in kwargs:
                    raise ValueError(f"{node.kind!r} node {nid!r}: input {port!r} not connected")

            if run is not None:
                # checked here, not on the way in: everything upstream has run by
                # now, so Stop takes effect at the next node that hasn't started.
                # A per-target stop shrinks "live" instead of flipping "cancel" —
                # nodes shared with an uncancelled target stay live and keep going.
                live = run.get("live")
                if run["cancel"] or (live is not None and nid not in live):
                    raise Cancelled("stopped")
                # this node's _total/_step report under its id
                run["node"] = nid
            result = inst.run(**kwargs)
        except (NodeError, Cancelled):
            raise  # already pinned to the upstream node that actually failed
        except Exception as e:
            raise NodeError(nid, str(e)) from e
        if not isinstance(result, dict):  # bare value -> first declared port (or display image)
            result = {cls.outputs[0] if cls.outputs else "image": result}
        cache[nid] = result
        if sig is not None:  # only reached on success — failures/cancels raised above
            _cache_put(sig, result)
        return result

    return node_outputs(target)
