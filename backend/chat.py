"""KumoFlow chat assistant — propose pipelines from a goal, explain the current graph.

One validated structured emission per turn (no tool loop): the model returns a
ChatTurn; an output validator rejects plans that fail validate_plan or arrive
before the brief (input_kind + output_handling) is complete, and Pydantic AI
retries with the errors fed back. catalog() and validate_plan() are plain
transport-agnostic functions so a future MCP server can wrap them — POST /chat
in main.py is one consumer.

How it all wires together (brief, prompt, gate, Ollama, VRAM choreography,
frontend) is documented in wiki/chat-assistant.md. Prompt, schemas and
validator were smoke-tested against live Ollama before this module existed.
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nodes import REGISTRY

CHAT_MODEL = os.environ.get("KUMOFLOW_CHAT_MODEL", "gpt-oss:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Not in the catalog and rejected in plans: custom is unsandboxed exec
# (v1 excludes code authoring), note is canvas-only prose with no ports.
EXCLUDED = {"custom", "note"}


# --- schemas -----------------------------------------------------------------


class PlanStep(BaseModel):
    kind: str
    config: dict[str, Any] = {}  # overrides of the node's defaults


class PlanWire(BaseModel):
    model_config = {"populate_by_name": True}

    from_: int = Field(alias="from")  # index into steps
    port: str  # source output port
    to: int
    target_port: str


class Requirement(BaseModel):
    text: str  # one clause of the user's ask, in their words
    steps: list[int]  # indices into plan.steps that satisfy it; [] = unsupported


class Plan(BaseModel):
    steps: list[PlanStep]
    wiring: list[PlanWire]
    # no default: the grammar-constrained decode must emit the coverage map —
    # the checklist effect is the point (wiki/request-alignment.md, phase 2)
    requirements: list[Requirement]


class ChatTurn(BaseModel):
    # reply stays the first field: field order is JSON order, and streaming a
    # partial reply (v1.5) needs it at the front
    reply: str  # question to the user OR plan explanation
    input_kind: Literal["image", "video"] | None
    output_handling: Literal["view", "export", "app"] | None
    plan: Plan | None = None


# --- catalog -----------------------------------------------------------------


def catalog(input_kind: str | None = None) -> str:
    """The node catalog for the system prompt, straight from the registry:
    kind, label, category, docstring, ports, config defaults. When input_kind
    is known, only nodes whose modalities include it (the palette's filter)."""
    lines = []
    for kind, cls in sorted(REGISTRY.items()):
        if kind in EXCLUDED:
            continue
        if input_kind and input_kind not in cls.modalities:
            continue
        s = cls.spec()
        doc = " ".join((cls.__doc__ or "").split())
        cfg = ", ".join(f"{k}={v!r}" for k, v in s["config"].items())
        lines.append(
            f"- {kind} ({s['label']}, {s['category']}): {doc}\n"
            f"  inputs={s['inputs']} required={s['required_inputs']} outputs={s['outputs']}"
            + (f" config: {cfg}" if cfg else "")
        )
    return "\n".join(lines)


# --- plan validation -----------------------------------------------------------


def _mate(src: str, tgt: str) -> bool:
    """Same rule as portsMate() in flow.ts: names equal, or image -> overlay."""
    return src == tgt or (src == "image" and tgt == "overlay")


def validate_plan(plan: Plan, input_kind: str, output_handling: str) -> list[str]:
    """Human-readable reasons this plan can't go on the canvas (empty = valid)."""
    errs = []
    for i, s in enumerate(plan.steps):
        if s.kind in EXCLUDED or s.kind not in REGISTRY:
            errs.append(f"step {i}: unknown or disallowed kind {s.kind!r}")
    if errs:
        return errs  # everything below needs real node classes
    classes = [REGISTRY[s.kind] for s in plan.steps]

    # at least one, not exactly one: auxiliary media (a logo for warp_overlay's
    # overlay port) legitimately arrives via a second load. Wrong-modality
    # loaders (load_video in an image brief) fall to the per-step modality check.
    want = {"image": "load", "video": "load_video"}[input_kind]
    if not any(s.kind == want for s in plan.steps):
        errs.append(f"plan must contain an input node of kind {want!r}")

    if output_handling == "export":
        if not any(s.kind == "export" for s in plan.steps):
            errs.append("output_handling 'export' requires an export step")
    elif not any(c.category == "Output" for c in classes):
        errs.append("plan must end in at least one Output-category node")

    for i, c in enumerate(classes):
        if input_kind not in c.modalities:
            errs.append(f"step {i} ({plan.steps[i].kind}) does not support {input_kind}")

    # coverage map: every clause of the request accounted for (mechanically —
    # whether the mapped steps really satisfy the clause is on the model)
    if not plan.requirements:
        errs.append("requirements is empty — break the user's request into clauses and map each to its steps")
    for i, r in enumerate(plan.requirements):
        if bad := [s for s in r.steps if not 0 <= s < len(plan.steps)]:
            errs.append(f"requirement {i} ({r.text!r}): step indices {bad} out of range")

    wired = set()  # (step, port) that received a wire
    adjacency: dict[int, list[int]] = {}
    for w in plan.wiring:
        if not (0 <= w.from_ < len(plan.steps) and 0 <= w.to < len(plan.steps)):
            errs.append(f"wire {w.from_}->{w.to}: step index out of range")
            continue
        adjacency.setdefault(w.from_, []).append(w.to)
        src, tgt = classes[w.from_], classes[w.to]
        if w.port not in src.outputs:
            errs.append(f"wire: {plan.steps[w.from_].kind} has no output port {w.port!r}")
        if w.target_port not in tgt.inputs:
            errs.append(f"wire: {plan.steps[w.to].kind} has no input port {w.target_port!r}")
        if not _mate(w.port, w.target_port):
            errs.append(f"wire: port {w.port!r} does not mate with {w.target_port!r}")
        wired.add((w.to, w.target_port))

    for i, c in enumerate(classes):
        for p in c.required_ports():
            if (i, p) not in wired:
                errs.append(f"step {i} ({plan.steps[i].kind}): required input {p!r} not wired")

    for i, s in enumerate(plan.steps):
        bad = set(s.config) - set(classes[i].model_fields)
        if bad:
            errs.append(f"step {i} ({s.kind}): unknown config keys {sorted(bad)}")
        else:
            try:
                REGISTRY[s.kind](**s.config)  # pydantic validates the values
            except Exception as e:
                errs.append(f"step {i} ({s.kind}): bad config: {e}")

    # acyclic (Kahn over the in-range wires)
    indeg = {i: 0 for i in range(len(plan.steps))}
    for tos in adjacency.values():
        for t in tos:
            indeg[t] += 1
    queue = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        seen += 1
        for t in adjacency.get(queue.pop(), ()):
            indeg[t] -= 1
            if indeg[t] == 0:
                queue.append(t)
    if seen < len(plan.steps):
        errs.append("plan wiring contains a cycle")

    if len(plan.steps) > 1:
        touched = {w.from_ for w in plan.wiring} | {w.to for w in plan.wiring}
        for i in range(len(plan.steps)):
            if i not in touched:
                errs.append(f"step {i} ({plan.steps[i].kind}) is not wired to anything")
    return errs


def turn_errors(turn: ChatTurn) -> list[str]:
    """Why a turn must be retried (empty = acceptable). The agent's output
    validator raises ModelRetry with these; factored out so it's testable
    without a model. A plan may only appear once the brief is complete."""
    # a plan pasted as JSON text into `reply` is invisible to the UI (the
    # apply button keys off the `plan` field) — seen in the wild, so gated
    if '"steps"' in turn.reply and '"wiring"' in turn.reply:
        return ["`reply` must be prose only — emit the plan in the `plan` field, never as JSON inside `reply`"]
    if turn.plan is None:
        return []
    if turn.input_kind is None or turn.output_handling is None:
        return ["brief incomplete: ask about the missing brief field instead of planning"]
    if errs := validate_plan(turn.plan, turn.input_kind, turn.output_handling):
        return ["invalid plan:"] + errs
    # steps: [] claims a clause is unsupported — only acceptable when the user
    # is being told so (the gate can check for *a* reply, not its content)
    if any(not r.steps for r in turn.plan.requirements) and not turn.reply.strip():
        return ["a requirement with steps: [] needs `reply` to explain what can't be done"]
    return []


# --- system prompt -------------------------------------------------------------

# Worked examples for RULES — real dicts, not prose, so tests can assert each
# one passes the gate: the prompt can never teach a shape the validator
# rejects. Two different shapes on purpose (one example is an anchor, not a
# library); the second is the motivating failure of wiki/request-alignment.md.
EXAMPLE_TURNS = [
    {
        "reply": "Load your images, Segment (SAM3) finds every person from the text prompt, View Image draws the masks.",
        "input_kind": "image",
        "output_handling": "view",
        "plan": {
            "steps": [
                {"kind": "load"},
                {"kind": "text_prompt", "config": {"text": "person"}},
                {"kind": "segment"},
                {"kind": "view"},
            ],
            "wiring": [
                {"from": 0, "port": "image", "to": 2, "target_port": "image"},
                {"from": 1, "port": "prompts", "to": 2, "target_port": "prompts"},
                {"from": 0, "port": "image", "to": 3, "target_port": "image"},
                {"from": 2, "port": "masks", "to": 3, "target_port": "masks"},
                {"from": 2, "port": "boxes", "to": 3, "target_port": "boxes"},
            ],
            "requirements": [{"text": "outline all people", "steps": [1, 2, 3]}],
        },
    },
    {
        "reply": "Segment finds the cats, Refine Masks inverts the selection to everything around them, "
        "Warp Overlay paints that white, View Image shows the cats on the white canvas.",
        "input_kind": "image",
        "output_handling": "view",
        "plan": {
            "steps": [
                {"kind": "load"},
                {"kind": "text_prompt", "config": {"text": "cat"}},
                {"kind": "segment"},
                {"kind": "mask_ops", "config": {"invert": True}},
                {"kind": "warp_overlay", "config": {"color": "#ffffff"}},
                {"kind": "view"},
            ],
            "wiring": [
                {"from": 0, "port": "image", "to": 2, "target_port": "image"},
                {"from": 1, "port": "prompts", "to": 2, "target_port": "prompts"},
                {"from": 2, "port": "masks", "to": 3, "target_port": "masks"},
                {"from": 0, "port": "image", "to": 4, "target_port": "image"},
                {"from": 3, "port": "masks", "to": 4, "target_port": "masks"},
                {"from": 4, "port": "image", "to": 5, "target_port": "image"},
            ],
            "requirements": [
                {"text": "detect cats", "steps": [1, 2]},
                {"text": "show them on a white canvas", "steps": [3, 4, 5]},
            ],
        },
    },
]

# built by concatenation, not .format(): the prose itself contains JSON braces
RULES = (
    """You are KumoFlow's assistant. KumoFlow is a visual computer-vision pipeline editor:
users wire nodes with typed ports on a canvas, then run the graph.

Fill input_kind and output_handling from the conversation. Values listed as ESTABLISHED are
settled - do not ask about them again. If either is still unknown, ask for exactly that one
thing in `reply` and set plan to null. Never emit a plan while either is unknown.
output_handling meanings: view = see results on canvas, export = download the images, app = a
simple end-user web page.

A plan is a self-contained subgraph: `steps` (kind + config overrides), `wiring`
({"from": step_index, "port": output_port, "to": step_index, "target_port": input_port}) and
`requirements`. One input node for the user's data (load for image, load_video for video); an
extra load step is allowed for auxiliary media the user mentions (e.g. a logo wired into
warp_overlay's overlay port). End in nodes that match output_handling (for "app" too, end in
view nodes). Prefer default config; only set what the user's goal needs. No coordinates. When
you emit a plan, `reply` briefly explains it
in prose — never print plan JSON inside `reply`; the UI can only apply what is in the `plan`
field.

Break the request into `requirements`: each is one clause of the user's ask (their words),
mapped to the indices of the steps that satisfy it. Never emit a plan that silently drops a
clause — a clause the available nodes can't express gets "steps": [] and an explanation of the
limitation in `reply`.

Transforms like blur act on the WHOLE image unless masks are wired into them. When the goal
targets only part of the image (an object, the background, faces), first get masks via
text_prompt -> segment; when the target is everything EXCEPT the named subject (e.g. "the
background"), segment the subject and pass its masks through mask_ops with invert=true.

Example turn (goal: outline all people; input_kind=image, output_handling=view established):
"""
    + json.dumps(EXAMPLE_TURNS[0])
    + """

Example turn (goal: detect cats and show them on a white canvas; input_kind=image,
output_handling=view established — the white-canvas clause gets its own requirement, satisfied
by mask_ops(invert) + warp_overlay(color)):
"""
    + json.dumps(EXAMPLE_TURNS[1])
)

# Recipes are single-sourced from AGENTS.md (repo rule: chat, CLI and MCP must
# agree by construction — mcp_server._instructions() embeds the same section).
_sections = (Path(__file__).resolve().parent.parent / "AGENTS.md").read_text().split("\n## ")
RECIPES = next((s for s in _sections if s.startswith("Recipes\n")), None)
assert RECIPES is not None, "AGENTS.md '## Recipes' section not found (renamed?) — the chat prompt embeds it"

PORTS = """Port types (an output mates with the same-named input; 'image' also plugs into 'overlay'):
image (batch of images), video (batch of clips), masks (instance masks), boxes (bounding boxes),
prompts (text/drawn prompts for models), labels (per-image class labels), classification (scores),
embedding (one vector per image), ids (track ids). Batches are lists; sizes may differ per item."""


def _scrub(v):
    """Config values as the prompt may show them — an uploaded base64 image
    must never reach the context window."""
    if isinstance(v, str) and (v.startswith("data:") or len(v) > 300):
        return f"<{len(v)} chars>"
    if isinstance(v, list):
        return [_scrub(x) for x in v]
    return v


def _compact_graph(graph: dict) -> dict:
    """The canvas graph as the prompt shows it: structure only — positions,
    labels and edge ids dropped, empty config values omitted, uploads scrubbed.
    Wire vocabulary matches the plan schema so the model reads one language."""
    return {
        "nodes": [
            {
                "id": n["id"],
                "kind": n["kind"],
                "config": {k: _scrub(v) for k, v in (n.get("config") or {}).items() if v not in ("", [], None)},
            }
            for n in graph.get("nodes", [])
        ],
        "edges": [
            {
                "from": e["source"],
                "port": e.get("sourceHandle") or "image",
                "to": e["target"],
                "target_port": e.get("targetHandle") or "image",
            }
            for e in graph.get("edges", [])
        ],
    }


def system_prompt(input_kind: str | None = None, output_handling: str | None = None, graph: dict | None = None) -> str:
    est = []
    if input_kind:
        est.append(f"input_kind={input_kind}")
    if output_handling:
        est.append(f"output_handling={output_handling}")
    parts = [RULES, PORTS, RECIPES, "AVAILABLE NODES:\n" + catalog(input_kind)]
    if est:
        parts.append("ESTABLISHED: " + ", ".join(est))
    if graph and graph.get("nodes"):
        parts.append(
            "CURRENT GRAPH (you may be asked to explain it; you cannot modify it — "
            "new plans are self-contained subgraphs):\n" + json.dumps(_compact_graph(graph))
        )
    return "\n\n".join(parts)


def derive_input_kind(graph: dict | None) -> str | None:
    """input_kind implied by the submitted graph's input node, if unambiguous."""
    kinds = {n.get("kind") for n in (graph or {}).get("nodes", [])}
    has = {k for k, kind in (("image", "load"), ("video", "load_video")) if kind in kinds}
    return next(iter(has)) if len(has) == 1 else None


# --- the model + agent ----------------------------------------------------------

# Context-length trap (verified, wiki/chat-assistant.md): Ollama's runtime default
# num_ctx=4096 silently truncates our ~5k-token prompt, and the OpenAI-compat
# /v1 endpoint ignores num_ctx in extra_body. So we chat with a derived model
# that pins num_ctx — created via the native API; it shares blobs (no copy).
NUM_CTX = 16384
_ENSURED: set[str] = set()


def model_name(base: str = CHAT_MODEL) -> str:
    return f"{base.removesuffix(':latest')}-kumoflow"


def _api_post(path: str, payload: dict, timeout: int) -> None:
    api = OLLAMA_BASE_URL.removesuffix("/").removesuffix("/v1")
    req = urllib.request.Request(
        f"{api}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        res.read()


def ensure_model(base: str = CHAT_MODEL) -> str:
    """Idempotently create the 16k-ctx '-kumoflow' variant of `base` and
    return its name. Raises if Ollama is unreachable or `base` isn't pulled."""
    if base not in _ENSURED:
        _api_post("/api/create", {"model": model_name(base), "from": base, "parameters": {"num_ctx": NUM_CTX}}, 30)
        _ENSURED.add(base)
    return model_name(base)


def unload_model() -> None:
    """Ask Ollama to drop the chat model from VRAM right now (the 12 GB card
    can't hold it and SAM3 together — /run calls this before evaluating, and
    /chat evicts nodes._MODELS as the mirror move). Raises if Ollama is down
    or the model was never created; callers treat that as nothing-to-evict."""
    _api_post("/api/generate", {"model": model_name(), "keep_alive": 0}, 10)


def chat_turn(
    messages: list[dict],
    graph: dict | None = None,
    input_kind: str | None = None,
    output_handling: str | None = None,
) -> ChatTurn:
    """One assistant turn: full history in (role/content dicts, last one the
    live user message), validated ChatTurn out. Raises on Ollama failure."""
    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.output import NativeOutput
    from pydantic_ai.providers.ollama import OllamaProvider

    if input_kind is None:
        input_kind = derive_input_kind(graph)

    # instructions=, not system_prompt=: instructions are re-sent with every
    # request even when message_history is passed — a history-carrying turn
    # would otherwise lose the catalog entirely.
    agent = Agent(
        OllamaModel(ensure_model(), provider=OllamaProvider(base_url=OLLAMA_BASE_URL)),
        output_type=NativeOutput(ChatTurn),  # grammar-constrained decoding; worked on every model smoke-tested
        instructions=system_prompt(input_kind, output_handling, graph),
        retries=2,  # 2.28: there is no output_retries; ModelRetry counts against this
        model_settings={"timeout": 480},
    )

    last: ChatTurn | None = None

    @agent.output_validator
    def gate(turn: ChatTurn) -> ChatTurn:
        nonlocal last
        last = turn
        if errs := turn_errors(turn):
            raise ModelRetry("\n".join(errs))
        return turn

    history = [
        ModelRequest(parts=[UserPromptPart(content=m["content"])])
        if m["role"] == "user"
        else ModelResponse(parts=[TextPart(content=m["content"])])
        for m in messages[:-1]
    ]
    try:
        return agent.run_sync(messages[-1]["content"], message_history=history).output
    except UnexpectedModelBehavior:
        if last is None:
            raise  # the model never produced a ChatTurn at all
        # retries exhausted on an invalid plan — never surface it
        return last.model_copy(
            update={
                "plan": None,
                "reply": last.reply
                + "\n\nI couldn't turn that into a pipeline that validates — try rephrasing your goal.",
            }
        )
