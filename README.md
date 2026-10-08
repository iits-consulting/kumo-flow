# KumoFlow

A low-code node based platform for computer vision workflows.

## Run it

```sh
make up      # backend http://localhost:8000, frontend http://localhost:5173 (logs: backend.log, frontend.log)
make down    # stop both
make test    # backend tests
```

Or by hand:

```sh
cd backend && uv run main.py        # http://localhost:8000
cd frontend && npm run dev          # http://localhost:5173
```

## Using it from a coding agent

Any coding agent can author and run workflows — [AGENTS.md](AGENTS.md) has the full
contract. Two ways in (backend must be running):

**CLI** — works with anything that can run shell commands:

```sh
cd backend
uv run cli.py catalog                    # machine-readable node list
uv run cli.py validate workflow.json    # exit 0 = runnable
uv run cli.py run workflow.json --out results/
```

**MCP server** — for agents wired in persistently. Register the stdio server,
e.g. for Claude Code:

```sh
claude mcp add kumoflow -- uv run --directory <repo>/backend --extra agent mcp_server.py
```

Any other MCP client takes the same command (`uv`) and args
(`run --directory <repo>/backend --extra agent mcp_server.py`). The server
exposes eight tools — `catalog`, `validate_workflow`, `run_workflow`,
`get_progress`, `stop_run`, `fetch_media`, `get_canvas`, `set_canvas` — all
delegating to the backend HTTP API, so it never loads models itself.

## How it works

See the [wiki](wiki/) for how the chat assistant, graph runs, structural signatures and the UI builder work.


## Adding a node

All node types live in `backend/nodes.py`. Subclass `InputNode`, `TransformNode`,
or `OutputNode` (or `Node` directly for multi-port nodes), set `kind` and `label`,
declare config as pydantic fields (with defaults), and implement `run`. The
frontend fetches the palette from `GET /nodes`, so the new node appears in the
UI automatically — no frontend changes needed.

Batches are first-class: an image batch is a list of `(C, H, W)` RGB uint8
torch tensors (a single image is just a length-1 list), so `run` takes and
returns lists. The full port contract is in the `nodes.py` module docstring.

```python
class Invert(TransformNode):
    kind = "invert"
    label = "Invert"

    def run(self, image):
        return [255 - img for img in image]
```
