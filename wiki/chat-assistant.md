# How the chat assistant works

The chat panel turns a goal ("blur the background of my images") into a
pipeline proposal, and answers questions about the current graph. It runs
against a **local Ollama model** and is built on one core decision:
**propose-then-insert**. The model never mutates the canvas — it returns a
plan, the chat renders it as a card, and the user inserts it with one click.
From then on it's a normal subgraph: autosave, undo, ▶ Run, all existing paths.

The second load-bearing decision: **one validated structured emission per
turn**. There is no agentic tool loop — local models are unreliable in them.
Every assistant turn is a single JSON object (`ChatTurn`), and all reliability
comes from schema + validation + retry, not from model obedience.

Everything lives in two places: `backend/chat.py` (schemas, catalog, validator,
prompt, agent) and `frontend/src/lib/ChatPanel.svelte` + `chat.ts` (panel,
state, insertion). `POST /chat` in `main.py` is a thin consumer — `catalog()`
and `validate_plan()` are plain transport-agnostic functions, so a future MCP
server can wrap them without touching HTTP.

## One turn, end to end

The backend is **stateless**: the frontend owns chat history, the graph, and
the brief, and sends all of it with every request.

```
POST /chat
{
  "messages": [{"role": "user"|"assistant", "content": str}, ...],  // last 20
  "graph":    {"nodes": [...], "edges": [...]},                     // the live canvas, may be empty
  "brief":    {"input_kind": ..., "output_handling": ...}           // last known, echoed back
}
→ 200: ChatTurn — {"reply", "input_kind", "output_handling", "plan"|null}
→ 502: Ollama unreachable / model missing (the UI shows the "is Ollama running?" hint)
```

`chat_turn()` builds a Pydantic AI agent per request, replays the history,
and returns the validated turn. A plan is steps + wiring only — **no
coordinates**; the frontend lays inserted nodes out with the same
`layoutNodes()` recipes use:

```python
class PlanStep(BaseModel):
    kind: str
    config: dict = {}          # overrides of the node's defaults

class PlanWire(BaseModel):
    from_: int                 # index into steps (JSON alias "from")
    port: str                  # source output port
    to: int
    target_port: str

class ChatTurn(BaseModel):
    reply: str                 # question to the user OR plan explanation
    input_kind: Literal["image", "video"] | None
    output_handling: Literal["view", "export", "app"] | None
    plan: Plan | None
```

`reply` is deliberately the first field: field order is JSON order, so
streaming a partial reply later (v1.5) stays possible.

## The brief: clarify before planning

The model must know two things before it may emit a plan: `input_kind`
(image or video) and `output_handling` (view / export / app). Enforcement is
**structural, not prose** — the fields sit in the response schema, and the
output validator rejects any turn that carries a plan while either is null.
If the user's first message pins both, the model plans in turn one; otherwise
its reply asks for exactly the missing field. The *how* (which nodes, in what
order) is never asked — figuring that out is the model's job.

Two prefills keep it from re-asking: the frontend echoes the last known brief
with every request, and the backend derives `input_kind` from the submitted
graph when it already contains a `load` / `load_video` node
(`derive_input_kind`). Established values are listed in the system prompt as
settled.

## The system prompt

Assembled fresh per request (`system_prompt()`), passed as `instructions=` —
not `system_prompt=` — because instructions are re-sent with every request
even when message history is passed; a history-carrying turn would otherwise
lose the catalog entirely. Its parts:

- **Rules** — what KumoFlow is, brief rules, plan rules, and the **masking
  rule**: transforms like blur act on the whole image unless masks are wired
  in; a goal targeting part of the image needs `text_prompt → sam3`, plus
  `mask_ops(invert=true)` when the target is everything *except* the named
  subject. Verified load-bearing: without it, every model tested blurred the
  whole image for "blur the background".
- **Ports** — ~10 hand-written lines distilling the port-type semantics from
  the `nodes.py` module docstring.
- **The catalog** — `catalog(input_kind)`, generated straight from
  `REGISTRY[kind].spec()` plus each node's docstring (that's why `spec()`
  carries `doc`). No hand-maintained node list anywhere: add a node to the
  registry and the assistant knows it. Excludes `custom` (unsandboxed exec —
  code authoring is out of v1) and `note`; when `input_kind` is known, only
  nodes whose `modalities` include it survive (the palette's own filter).
  Full catalog ≈ 5k tokens.
- **The current graph**, when non-empty — compacted (`_compact_graph`):
  positions, labels and edge ids dropped, empty config omitted, uploads
  scrubbed (`_scrub` — a base64 image must never reach the context window),
  and wires renamed to the plan's own vocabulary so the model reads one wiring
  language. Tagged "you may be asked to explain it; you cannot modify it".
- **One few-shot example** — a complete worked turn. Small local models need
  exactly one; keep it to one.

## The validation gate

The agent runs with `output_type=NativeOutput(ChatTurn)` — Ollama
grammar-constrained decoding, so the *shape* is always right — and an output
validator that decides whether the *content* is acceptable (`turn_errors`):

1. Plan JSON pasted into `reply` → rejected (seen in the wild: the apply
   button keys off the `plan` field, so a plan in prose is invisible to the UI).
2. Plan present but brief incomplete → "ask about the missing field instead".
3. `validate_plan()` errors → fed back verbatim.

`validate_plan()` is the whole safety story for insertion — everything a plan
could get wrong, each check a few lines: kinds exist and aren't excluded;
exactly one input node matching `input_kind`; sinks match `output_handling`
(`export` needs an `export` step, otherwise ≥1 Output-category node); every
step supports the modality; wire indices in range, ports exist, ports mate
(same rule as `portsMate()` in `flow.ts`: names equal, or `image → overlay`);
required ports wired; config keys ⊆ the node's fields and
`REGISTRY[kind](**config)` constructs (pydantic validates values for free);
acyclic; no disconnected steps.

Rejection raises `ModelRetry`, so Pydantic AI feeds the errors back and the
model gets another go (`retries=2`). If retries are exhausted, the turn is
returned with `plan` stripped and a "couldn't validate, try rephrasing"
sentence appended — **an invalid plan is never surfaced**. The smoke tests
showed this loop repairing real failures (gemma4 needed one retry, and
`qwen3:0.6b`'s schema-valid garbage was rejected every time — the gate works
as a negative control too).

## Ollama wiring

Config via env vars so a later vLLM move is a URL swap:
`KUMOFLOW_CHAT_MODEL` (default `gpt-oss:latest`), `OLLAMA_BASE_URL`
(default `http://localhost:11434/v1`). Pydantic AI talks to Ollama's
OpenAI-compatible endpoint.

**The context-length trap** (verified on Ollama 0.32.5): Ollama's runtime
default is `num_ctx=4096` and it *silently truncates* longer prompts — a WARN
in the server log is the only trace. Our system prompt is ~5.3k tokens, so
without a fix the model never sees most of the catalog. The `/v1` endpoint
ignores `num_ctx` in `extra_body`, so it can't be set per request. The fix
(`ensure_model()`): at startup — and again on first `/chat` if Ollama was down
— the backend idempotently creates a derived model via the native API
(`POST /api/create`, `{"from": base, "parameters": {"num_ctx": 16384}}`).
It shares blobs with the base model (no copy, instant) and all chatting goes
through the `<base>-kumoflow` variant.

**VRAM choreography** — a 12 GB card can't hold the chat LLM (~10.5 GB
resident) and SAM3 at once, so the two sides evict each other:

- `POST /chat` first calls `nodes.free_models()` — drops the cached torch
  models and empties the CUDA cache.
- `POST /run` first calls `chat.unload_model()` — `keep_alive: 0` via the
  native API, dropping the chat model from VRAM immediately instead of after
  Ollama's 5-minute default.

Both are unconditional and best-effort; a chat and a run in flight
concurrently tug at each other, but nothing breaks — a model a running node
still references lives until that reference dies, it just frees late.

## The frontend

`ChatPanel.svelte` mirrors the left palette on the right edge: collapsed by
default, open state and the conversation (messages + brief) persisted to
localStorage, a Clear button that wipes both. The start page's "…or describe
what you want to build" button opens it and focuses the input.

Two wire-format details matter:

- **Plans are echoed back into history.** The API only carries
  `{role, content}`, so when the frontend sends an assistant message that
  carried a plan, it appends `[my proposed plan: <JSON>]` to the content.
  Without this the model can't refine its own proposal ("add a resize before
  the blur") — the prose reply alone doesn't say what it built.
- **Stop is client-side only.** A pending turn can be aborted
  (`AbortController`), which frees the input immediately; the backend finishes
  its turn server-side and the response is discarded. Local models can grind
  for minutes — the request timeout is 480 s.

A turn that carries a plan renders a card under the reply: the steps as
palette labels joined with arrows, plus **Add to canvas** (never
auto-inserted; once clicked it flips to "Added ✓"). Insertion
(`insertPlan()` in `chat.ts`) is the recipe path re-used: `createNode` per
step with config overrides applied, edges from the wiring, `layoutNodes()` for
positions, the subgraph translated to the same origin `addRecipe` would pick.
For `output_handling === "app"` an App node lands beside the subgraph with a
section per `appEligible` node — the plan itself still just ends in view
nodes; the app wrapper is a frontend concept (see
[ui-builder.md](ui-builder.md)).

## Model choice

Smoke-tested on the dev machine (RTX 5070 Ti Laptop, 12 GB VRAM), full
catalog + validator + retry gate against live Ollama:

- **`gpt-oss` (20B MoE)** — the default. All cases pass in 7–33 s/turn;
  produced the semantically correct masked-blur pipeline first try.
- **`gemma4`** — the fallback. All pass (4–48 s); asked a sensible clarifying
  question when the subject was unnamed.
- **`qwen3.6:27b`** — unusable on 12 GB (CPU/GPU split, ~8 min/turn); only
  worth selecting on ≥24 GB GPUs.

## Testing

`backend/test_chat.py` covers the deterministic parts with no LLM calls:
catalog exclusions and modality filtering, one test per `validate_plan`
failure class, and the turn-gating rules. Live-model behavior is covered by
manual acceptance instead:

1. Empty canvas → "I have images and want the background blurred." →
   `input_kind` fills from the message; the assistant asks how to handle the
   output — it does not plan yet.
2. "Just view them" → plan card ≈ the Blur Background recipe; Add to canvas
   inserts a laid-out, fully wired subgraph; ▶ Run works on it unmodified.
3. Same conversation ending in "as an app" → subgraph plus an App node with
   sections for the load and view nodes.
4. Canvas with an existing pipeline → "what does this graph do?" → correct
   prose explanation, no plan card.
5. Stop Ollama → send → error bubble with the hint, app otherwise unaffected;
   retry works after `ollama serve` comes back.
