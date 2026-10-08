# How the UI builder works

The **App node** turns a working canvas graph into an end-user web page: pick
a few nodes off the canvas, get a page with upload widgets on top and result
panes below, and put it on its own port with one click. One page spec gets
rendered three times by the same components —

1. **the node card** (`frontend/src/lib/AppNode.svelte`) — where you compose it,
2. **the fullscreen overlay** (▶ View) — a live preview against the editor's
   own state,
3. **the deployed standalone page** — a static Svelte bundle served by a
   `deploy.py` child process, no editor and no `@xyflow` anywhere in it.

## A node with no ports

The App node is a normal palette entry (`APP_ITEM`, `frontend/src/lib/flow.ts`)
— but the only one *not* served by the backend; the frontend appends it after
fetching the real palette. It has `inputs: []`, `outputs: []` and xyflow type
`"app"`, so everything that filters on `type === "flow"` — backend payloads,
`displayIds`, auto-wiring — ignores it for free. `backend/nodes.py`'s
`REGISTRY` has no `"app"` entry: **the backend never learns what an App node
is.** Until deploy time it is a purely frontend concept riding along in the
graph JSON, persisted wherever the rest of the canvas is (localStorage
autosave, downloaded `workflow.json`).

Its data is a page spec by reference, not by copy:

```ts
interface AppSection {
  node: string;          // referenced canvas node id
  label?: string;        // empty falls back to the node's own label
  description?: string;
}
interface AppNodeData {
  title: string;
  sections: AppSection[];
  deployment?: string;   // name of its live deployment, set on deploy
}
```

The card itself is a small form: a title input and an "+ Add section…" dropdown
listing every eligible node not already used. Eligible (`appEligible`,
`flow.ts`) means an **input node** (`load`, `load_video` — these become upload
widgets) or a **display node** (no outputs; `custom` and `note` excluded).
Sections reorder with ↑/↓, and each takes an optional label override and
description. Because a section stores only a node id, deleting the referenced
canvas node doesn't corrupt anything — the card shows a "missing node" pill and
the page silently drops the section.

## Preview: the same page against the live canvas

▶ View calls `appPage(nodeId)` (`frontend/src/App.svelte`), which resolves the
spec against the live canvas on every read: dead sections dropped, empty labels
filled from the referenced node's label, and `displayIds` — the non-input
sections — doubling as the page's run targets. The overlay renders
`render/AppView.svelte` against the *editor's own* results/errors/progress
stores, so the preview always matches unsaved edits. (It also grows out of the
node's on-screen rect via a computed `transform-origin` — cosmetic, but that's
why opening it feels anchored to the node.)

## One renderer, two hosts

Before this commit, `FlowNode.svelte` rendered every result inline —
classification bars, count table, scatter plot, image/video pagers, export
link, file pickers, with the pager markup duplicated four times. The commit
extracts all of it into `frontend/src/lib/render/`:

| Component | Renders |
|---|---|
| `Pager.svelte` | the shared "‹ i / n ›" control |
| `ImageView.svelte` / `VideoView.svelte` | paged image preview / clip player |
| `ClassificationView.svelte` | SigLIP bars + normalized/raw toggle |
| `CountView.svelte` | count table or big single number |
| `ScatterView.svelte` | embedding scatter + hover thumbnail |
| `ExportView.svelte` | zip download link |
| `UploadWidget.svelte` | Choose File / Choose Folder pickers |
| `AppView.svelte` | the app page layout itself (sections top-to-bottom) |
| `types.ts` | `NodeResult` & friends, plus `toNodeResult(raw, apiPrefix)` |

`FlowNode.svelte` (canvas card) and `AppView.svelte` (app page) both import the
same leaf components and feed them the same plain-data shapes — neither
composes the other. The load-bearing constraint is that **nothing under
`render/` imports `@xyflow/svelte`** (documented at the top of `types.ts`):
that's what lets the deployed bundle exist without dragging the whole canvas
along. It's a convention enforced by comment, not by lint.

## Deploy: baking the spec

The node's Deploy button opens the deploy dialog and, on confirm, `deployNow()`
(`App.svelte`) posts

```jsonc
{
  "graph": { /* backendGraph(), targets = the page's displayIds */ },
  "ui":    { "title": "...", "sections": [{ "node", "label", "description" }] },
  "name":  "my-app",
  "port":  8001
}
```

Two deliberate asymmetries in what gets baked:

- **Labels are baked.** `appPage()`'s live label fallback is flattened into the
  spec, so the deployed page never needs canvas node labels.
- **Kinds are not.** Each section's `kind` is stripped client-side and
  re-derived at request time from the artifact's own node list (`/info`'s
  `kinds` map) — the renderer choice always agrees with the graph that will
  actually run.

`POST /deploy` (`backend/main.py`) treats `ui` as an **opaque dict** — no
pydantic model, never validated, never part of `Graph`. It writes the artifact
to `backend/deployments/<name>.json` (graph snapshot with `ui` as a sibling
key), then spawns `uv run deploy.py deployments/<name>.json` as a child
process with `PORT` in its env — every deployment is its own OS process on its
own port, with its own model copies. A module-level `PROCS` registry tracks
them, and three rules fall out of it:

- **Same name again** → the old process is terminated and replaced in place.
- **Different name, same port, still alive** → `409` naming the holder.
- **`DELETE /deployments/{name}`** → terminate and forget, but the artifact
  file stays on disk — restartable any time.

`GET /deployments` reports `{name, port, has_ui, alive}` per child, with
`alive` polled live so a crashed child shows honestly. The Toolbar's "Live (N)"
popover is a thin view over this list. An `atexit` hook terminates every child
when the main backend exits.

## The deployed page

`deploy.py`'s `create_app` reads the artifact and immediately
`spec.pop("ui", None)` — the execution side (`Graph(**spec)`) never sees page
metadata. What it serves:

- `GET /info` — `{workflow, inputs, targets, kinds, ui}`. (This used to be the
  bare `GET /` in the first deploy commit; `client.py` was updated to match.)
- `GET /` — **only if `ui` is present**: the static `frontend/dist/app.html`,
  with `/assets` mounted alongside. An API-only deploy has no `/` at all
  (404). If the dist is missing, `create_app` fails fast with "run `npm run
  build`" — `vite.config.ts` builds two entry points, `index.html` (editor)
  and `app.html` (app page), into the same `dist/`.
- `POST /run`, `GET /media/{name}` — unchanged from the API-only deploy.

`app.html` is mounted by `src/app-main.ts` → `render/AppStandalone.svelte`,
which boots with one `GET /info`, filters out sections whose node id vanished
from the snapshot, and renders `AppView` — upload widgets for input sections,
result panes for the rest. Run is a single blocking `POST /run` (no progress
polling): a multipart form keyed by input-node id, where **no files means
"re-run on the deployed inputs"** baked into the artifact's config.

Server-side, each run rebuilds a fresh `Graph(**spec)` so one request's input
override can't leak into the next, decodes uploads into base64 `data:` URLs
overwriting that input node's `config["data"]` (a flagged `ponytail:` shortcut
— spill to disk paths if huge videos hurt), and calls `main.run_graph` in a
threadpool — the exact same `evaluate`/`_sigs`/result-`CACHE` path as the
editor's `/run` (see [graph-evaluation.md](graph-evaluation.md)). The response
is shaped by the same `_shape`, and the page maps results back onto sections by
node id.

## The shape of it

- **One spec, one renderer, three hosts.** Node card to edit, overlay to
  preview live, standalone bundle when deployed — all reading the same
  `AppSection[]` and drawing with the same `render/` components.
- **Indirection while editing, baked at deploy.** On the canvas a section is
  just a node id — labels, kinds, and results resolve live. Deploy flattens
  labels into the artifact and leaves kinds to `/info`, so the frozen page and
  the frozen graph can't disagree.
- **The backend's entire knowledge of "apps" is `ui: dict | None`.** No node
  kind, no schema, no coupling — the App node stays deletable without touching
  Python.

## Limits

- **Deployments don't survive a backend restart.** `PROCS` is in-memory and
  the `atexit` hook kills the children; only the artifact JSONs persist. After
  a restart everything is one click from being live again, but nothing comes
  back by itself.
- **The port-conflict check only scans `PROCS`.** A port held by anything else
  (a hand-launched `deploy.py`, some unrelated service) isn't detected — the
  child just dies on bind, and `alive: false` in `GET /deployments` is the
  only symptom.
- **Deployed apps bind `0.0.0.0` with no auth** on `/run` — reachable
  off-loopback by design, so anyone who can reach the port can run the
  pipeline and upload inputs.
- **Uploads ride through base64** on the deployed page, videos included — fine
  for images, a known ceiling for large videos (the editor's own upload path
  sends videos multipart to `/upload` precisely to avoid this).
- **`ui` is never validated.** A malformed spec deploys fine and only
  misbehaves in the browser; a stray top-level `"ui"` key in a hand-written
  artifact is silently consumed as page metadata.
