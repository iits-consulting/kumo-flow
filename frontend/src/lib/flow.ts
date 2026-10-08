import { MarkerType, type Node, type Edge } from "@xyflow/svelte";

export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Shared via Svelte context so nodes can trigger a backend run and read results.
export const FLOW_CTX = Symbol("flow");
// Result types live with the shared renderers (render/types.ts, xyflow-free);
// re-exported here so canvas-side imports keep one entry point.
export type { ClassResult, CountResult, ExportResult, NodeResult, VolumeResult } from "./render/types";
import type { ClassResult, CountResult, ExportResult, VolumeResult } from "./render/types";

// One detected object on a View node's legend: its color key (label / track id /
// instance index), the hex it's currently drawn in, and a thumbnail crop.
export interface LegendEntry {
  key: string;
  color: string;
  thumb?: string;
}

// One live deployment as GET /deployments reports it.
export interface Deployment {
  name: string;
  port: number;
  has_ui: boolean;
  alive: boolean;
}

export interface FlowContext {
  // Run the graph up to `targetId` (default: the node itself); results/errors
  // land under `nodeId`, so a node can display an upstream node's output.
  run: (nodeId: string, targetId?: string) => void;
  // Every display node on the canvas, in one backend pass (shared cache: a
  // SAM3 feeding two viewers runs once, not twice).
  runAll: () => void;
  // Cancel what's in flight — the backend gives up at the next work unit.
  // With a nodeId, only that node's target is stopped: upstream work shared
  // with other running targets keeps going (per-target stop).
  stop: (nodeId?: string) => void;
  // nodeId -> [work units done, total], while the run reports them. Only nodes
  // that can count their work (models, tracking, video decode) show up here.
  progress: Record<string, [number, number]>;
  // Stop was pressed but the run hasn't returned yet (a model chunk in flight).
  readonly stopping: boolean;
  // Source node id wired into (nodeId, port), or null if unconnected.
  upstream: (nodeId: string, port: string) => string | null;
  // The + button on an edge (FlowEdge): open the insert-a-node-between menu.
  // `mid` is the edge's midpoint in flow coords — where the new node lands.
  insertOnEdge: (edgeId: string, mid: { x: number; y: number }, event: MouseEvent) => void;
  // True if any palette node can be inserted on an edge carrying src into
  // (targetId, tgt) — FlowEdge hides its + button when nothing could be offered.
  canInsert: (src: string, targetId: string, tgt: string) => boolean;
  results: Record<string, string[]>; // nodeId -> one preview URL per batch item (data URL for scatter thumbnails)
  // nodeId -> the [w, h] each preview was downscaled from. Previews are display
  // size, so anything drawn on one (Crop's box, a prompt node's boxes/points)
  // has to be mapped through this to come out in source pixels.
  sizes: Record<string, [number, number][]>;
  // nodeId -> true flattened frame index per preview item, only when the
  // backend capped a video-flattened preview (PREVIEW_MAX_FRAMES) — absent or
  // shorter than `results[nodeId]` means preview index == true frame index.
  // Visual Prompt's per-frame drawings key off the true index (see FlowNode).
  frameIndices: Record<string, number[]>;
  classifications: Record<string, ClassResult[]>; // nodeId -> one ClassResult per batch item
  counts: Record<string, CountResult>; // nodeId -> per-image + total object counts
  downloads: Record<string, ExportResult>; // nodeId -> built export zip (Export node)
  points: Record<string, number[][]>; // nodeId -> one [x, y] scatter point per batch item
  videos: Record<string, string[]>; // nodeId -> one streamable /media webm URL per clip (View Video)
  // nodeId -> the objects a View node detected, one entry per color key (label /
  // track id / instance index). Rendered as pick-a-color chips under the result;
  // picks land in the node's `colors` config and apply on the next run.
  legends: Record<string, LegendEntry[]>;
  legendMore: Record<string, number>; // nodeId -> count of detected objects past the legend's chip cap
  volumes: Record<string, VolumeResult[]>; // nodeId -> one raw grayscale volume (+ optional seg overlay) per clip (View Volume)
  errors: Record<string, string>;
  running: Record<string, boolean>;
  // Per-node config edits, owned by App (keyed by node id) so they survive xyflow re-renders.
  // Overlaid onto the node's config when the graph is sent to the backend.
  inputs: Record<string, Record<string, unknown>>;
  // Count of in-flight runs whose computed subgraph includes this node id (see
  // `ancestors`). >0 means the node is currently being computed by the backend.
  activeNodes: Record<string, number>;
  // Bumped when a run fails because this prompt node is empty — instead of an
  // error, the node opens its input UI (text field / image to draw on).
  attention: Record<string, number>;
  // App node buttons, handled by App.svelte: open the app page fullscreen /
  // open the deploy dialog prefilled from the node.
  openApp: (nodeId: string) => void;
  deployApp: (nodeId: string) => void;
  // Live deployments (synced from GET /deployments) — App nodes match their
  // own by stored name; stopDeployment is the node's (and toolbar's) Stop.
  deployments: Deployment[];
  stopDeployment: (name: string) => void;
}

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  kind: string;
  color: string;
  inputs: string[]; // input port names (edge targetHandle)
  outputs: string[]; // output port names (edge sourceHandle)
  required_inputs: string[]; // subset of inputs that must be connected
  config: Record<string, unknown>;
  config_info?: Record<string, string>; // per-field help — settings panel shows an info icon
  config_options?: Record<string, ConfigOption[]>; // per-field dropdown presets
  config_hf?: Record<string, string[]>; // per-field HF Hub pipeline tags — Custom… input searches the Hub
}

// A settings-dropdown preset (backend Field json_schema_extra "options"): a
// plain string, or {value, label} — dict options also get a Custom… free-text
// escape hatch in the settings panel.
export type ConfigOption = string | { value: string; label: string };

// One palette entry per node class defined in backend/nodes.py.
export interface PaletteItem {
  kind: string;
  label: string;
  category: string; // palette grouping
  doc?: string; // node docstring — tooltip + search corpus
  color: string;
  inputs: string[];
  outputs: string[];
  required_inputs: string[];
  // input kinds ("image" / "video") this node is relevant for — inputs name
  // what they produce, processing nodes what they consume (see App's filter)
  modalities: string[];
  config: Record<string, unknown>;
  config_info?: Record<string, string>;
  config_options?: Record<string, ConfigOption[]>;
  config_hf?: Record<string, string[]>;
}

// Handle color per port data type — makes the typed dataflow visible: an
// `image` handle only mates with another `image` handle, etc. Unknown ports
// fall back to slate. Keep in sync with the port names in backend/nodes.py.
// One clearly-distinct hue per data type — no two ports may share a color
// family (bright yellow vs dark orange, blue vs emerald, pink vs violet all
// differ in hue AND lightness), so check both when adding a port type.
// Key order = the canonical port order (PORT_ORDER in backend/nodes.py):
// the custom-node port picker and its port lists follow it.
export const PORT_COLORS: Record<string, string> = {
  image: "#2563eb", // blue
  video: "#dc2626", // red — pure hue, clearly apart from boxes' magenta-pink
  overlay: "#2563eb", // a second image input (Warp Overlay's logo) — same type, same color
  prompts: "#facc15", // yellow
  embedding: "#0d9488", // teal — green-leaning so it can't be confused with image blue
  classification: "#ea580c", // orange (matches Classify nodes)
  boxes: "#db2777", // pink — per-image or per-frame; the value's structure tells which (backend _per_frame)
  masks: "#7c3aed", // violet — same: one port type covers image and video masks
  labels: "#65a30d", // lime
  ids: "#22c55e", // vivid green — the free hue gap; bright and saturated where embedding's teal is dark and muted
};
export const portColor = (name: string) => PORT_COLORS[name] ?? "#64748b";

// An output mates with the same-named input; `overlay` is an image input under
// a different role name, so image outputs plug into it too. Missing handle ids
// default to "image" (see backend Edge model).
export function portsMate(source: string | null | undefined, target: string | null | undefined): boolean {
  const s = source ?? "image";
  const t = target ?? "image";
  return s === t || (s === "image" && t === "overlay");
}

// Inline style for an xyflow Handle: colored by data type; required inputs and
// all outputs are filled, optional inputs are hollow (white with a colored ring).
export function handleStyle(port: string, filled: boolean): string {
  const c = portColor(port);
  const fill = filled ? c : "#fff";
  return `width:11px;height:11px;border-radius:50%;background:${fill};border:2px solid ${c};`;
}

// Edge props: the wire and its (direction-showing) arrowhead take the color of
// the ports they join, so a glance at a wire says what flows through it.
// Missing handle id = "image", same default as the backend Edge model.
export function edgeStyle(sourceHandle: string | null | undefined) {
  const c = portColor(sourceHandle ?? "image");
  return { style: `stroke:${c};stroke-width:2`, markerEnd: { type: MarkerType.ArrowClosed, color: c } };
}

export async function fetchPalette(): Promise<PaletteItem[]> {
  const res = await fetch(`${API}/nodes`);
  if (!res.ok) throw new Error(`GET /nodes failed: ${res.status}`);
  return res.json();
}

const readAsDataURL = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });

// Picked files -> a load/load_video node's config overlay ({data, name,
// labels}), shared by the Load nodes and the editor-hosted app page. Handles a
// single file, a multi-select, or a whole folder (webkitdirectory): a lone
// file stays B=1 (a plain string). A folder in the class-per-subfolder dataset
// layout (PetImages/Cat/1.jpg) also yields per-item labels: the immediate
// parent dir inside the picked folder; files at its root stay unlabeled ("").
export async function filesToInput(files: File[], media: "image" | "video" | "volume"): Promise<Record<string, unknown>> {
  const labels = files.map((f) => {
    const parts = f.webkitRelativePath.split("/");
    return parts.length >= 3 ? parts[parts.length - 2] : "";
  });
  let data: string[];
  if (media !== "image") {
    // videos and volumes go multipart to the backend once and the graph
    // carries the returned server paths — base64-in-JSON breaks past ~400MB
    // (V8's max string length) and would re-send the whole payload on every run
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`${res.status}`);
    data = (await res.json()).paths;
  } else {
    data = await Promise.all(files.map(readAsDataURL));
  }
  return {
    data: data.length === 1 ? data[0] : data,
    name: files.length === 1 ? files[0].name : `${files.length} ${media}s`,
    labels: labels.some(Boolean) ? labels : [],
  };
}

// --- The App node (kind "app", frontend-only) ------------------------------
// Composes an end-user page out of nodes already on the canvas. Its xyflow
// type is "app", so everything that filters on type === "flow" — backend
// payloads, displayIds, auto-wiring — ignores it for free.

export interface AppSection {
  node: string; // referenced canvas node id
  label?: string; // end-user text; empty falls back to the node's label
  description?: string;
}

export interface AppNodeData extends Record<string, unknown> {
  title: string;
  sections: AppSection[];
  deployment?: string; // name of its live deployment, set on deploy
}

// The one palette entry not served by the backend — appended after fetch.
export const APP_ITEM: PaletteItem = {
  kind: "app",
  label: "App",
  category: "Output",
  color: "#0ea5e9",
  inputs: [],
  outputs: [],
  required_inputs: [],
  modalities: ["image", "video"], // never filtered out by the canvas modality gate
  config: {},
};

// What an app page can render a face for: upload widgets for the input kinds,
// result panes for display nodes (no outputs; custom's empty outputs mean
// "none picked yet" and a note is canvas-only prose).
export const APP_INPUT_KINDS: Record<string, "image" | "video" | "volume"> = { load: "image", load_video: "video", load_volume: "volume" };
export const appEligible = (d: FlowNodeData) =>
  d.kind in APP_INPUT_KINDS || (!d.outputs.length && d.kind !== "custom" && d.kind !== "note");

// True if adding source→target would close a cycle, i.e. target already reaches source.
export function createsCycle(source: string, target: string, edges: Edge[]): boolean {
  if (source === target) return true;
  const out: Record<string, string[]> = {};
  for (const e of edges) (out[e.source] ??= []).push(e.target);
  const seen = new Set<string>();
  const stack = [target];
  while (stack.length) {
    const n = stack.pop()!;
    if (n === source) return true;
    if (seen.has(n)) continue;
    seen.add(n);
    stack.push(...(out[n] ?? []));
  }
  return false;
}

// All node ids that feed into `targetId`, transitively, including `targetId`
// itself — the subgraph the backend actually computes for a run targeting it.
export function ancestors(targetId: string, edges: Edge[]): Set<string> {
  const upstream: Record<string, string[]> = {};
  for (const e of edges) (upstream[e.target] ??= []).push(e.source);
  const seen = new Set<string>([targetId]);
  const stack = [targetId];
  while (stack.length) {
    const id = stack.pop()!;
    for (const src of upstream[id] ?? []) {
      if (!seen.has(src)) {
        seen.add(src);
        stack.push(src);
      }
    }
  }
  return seen;
}

// Fallback footprint of a node card before it's been measured — used for
// collision checks and group sizing.
const NODE_W = 220;
const NODE_H = 160;
export const nodeW = (n: Node) => n.measured?.width ?? n.width ?? NODE_W;
export const nodeH = (n: Node) => n.measured?.height ?? n.height ?? NODE_H;

// Absolute canvas position of a node — grouped children store positions
// relative to their parent group. One level only: groups can't be nested
// (groupSelected only takes top-level nodes).
export function absolutePosition(n: Node, nodes: Node[]): { x: number; y: number } {
  const parent = n.parentId ? nodes.find((m) => m.id === n.parentId) : undefined;
  return parent ? { x: parent.position.x + n.position.x, y: parent.position.y + n.position.y } : n.position;
}

// Where to drop a newly added node: to the right of `anchor` (the selected or
// most-recently-added node), bumped down until it clears existing nodes.
// With no anchor (empty canvas), falls back to the original cascade.
export function nextPosition(anchor: Node | undefined, existing: Node[]): { x: number; y: number } {
  if (!anchor) {
    const n = existing.length;
    return { x: 380 + (n % 4) * 40, y: 40 + n * 30 };
  }
  const a = absolutePosition(anchor, existing);
  const pos = { x: a.x + nodeW(anchor) + 60, y: a.y };
  const overlaps = (n: Node) => {
    const p = absolutePosition(n, existing);
    return pos.x < p.x + nodeW(n) && p.x < pos.x + NODE_W && pos.y < p.y + nodeH(n) && p.y < pos.y + NODE_H;
  };
  while (existing.some(overlaps)) pos.y += NODE_H + 20;
  return pos;
}

// Svelte Flow "subflow": a `type: "group"` node that child nodes reference via
// `parentId` + `extent: "parent"`, which clips/drags them together. Padding
// around the bounding box, plus a header strip reserved for the group's label.
const GROUP_PAD = 30;
const GROUP_HEAD = 34;

// Wrap `subset` in a new group node sized to its bounding box, re-parenting
// each child with a position relative to the group's own top-left corner
// (xyflow requires child positions be relative to their parent).
export function makeGroup(subset: Node[], label: string): { group: Node; children: Node[] } {
  const x1 = Math.min(...subset.map((n) => n.position.x));
  const y1 = Math.min(...subset.map((n) => n.position.y));
  const x2 = Math.max(...subset.map((n) => n.position.x + nodeW(n)));
  const y2 = Math.max(...subset.map((n) => n.position.y + nodeH(n)));
  const ox = x1 - GROUP_PAD;
  const oy = y1 - GROUP_PAD - GROUP_HEAD;
  const group: Node = {
    id: `group-${++seq}`,
    type: "group",
    position: { x: ox, y: oy },
    data: { label },
    // behind all flow nodes: a box spanning far-apart members would otherwise
    // cover (and steal clicks from) unrelated nodes caught in its bounding box
    zIndex: -1,
    // numeric width/height (not a style string) so layoutNodes can read the size back
    width: x2 - x1 + GROUP_PAD * 2,
    height: y2 - y1 + GROUP_PAD * 2 + GROUP_HEAD,
  };
  const children = subset.map((n) => ({
    ...n,
    parentId: group.id,
    extent: "parent" as const,
    position: { x: n.position.x - ox, y: n.position.y - oy },
  }));
  return { group, children };
}

// Group the selected top-level flow nodes (2+; group boxes excluded so groups
// can't nest) into a new group node. No-op if fewer than 2 qualify. Parent
// must precede its children in the returned array for xyflow to resolve their
// relative positions.
export function groupSelected(nodes: Node[], label = "Group"): Node[] {
  const selected = nodes.filter((n) => n.selected && n.type === "flow" && !n.parentId);
  if (selected.length < 2) return nodes;
  const { group, children } = makeGroup(selected, label);
  const ids = new Set(selected.map((n) => n.id));
  return [...nodes.filter((n) => !ids.has(n.id)), group, ...children];
}

// The reverse: dissolve any selected group boxes, returning their children to
// the top level at their absolute canvas positions.
export function ungroupSelected(nodes: Node[]): Node[] {
  const dissolved = new Set(nodes.filter((n) => n.type === "group" && n.selected).map((n) => n.id));
  if (!dissolved.size) return nodes;
  return nodes
    .filter((n) => !dissolved.has(n.id))
    .map((n) => {
      if (!n.parentId || !dissolved.has(n.parentId)) return n;
      const { parentId, extent, ...rest } = n;
      return { ...rest, position: absolutePosition(n, nodes) };
    });
}

// Roboflow-style auto-wire for a newly added node, in both directions around
// the anchor (the selected/last-added node):
//  1. each of the new node's input ports pulls from the anchor if it has a
//     matching output, else from the most-recently-added other node that does
//     — so a multi-input node (e.g. SAM3: image + prompts) gets wired from
//     wherever its sources actually are, not just the immediate anchor.
//  2. any of the anchor's own inputs still unconnected get filled from the new
//     node's outputs — so adding a Prompt after a Prompt-less SAM3 wires into
//     SAM3's open port, regardless of add order.
// Single hop only (no transitive search); createsCycle guards each edge since,
// with two directions in play, one auto-added edge can enable a cycle for another.
export function autoConnectEdges(
  item: PaletteItem,
  newNodeId: string,
  anchor: Node | undefined,
  existing: Node[],
  edges: Edge[],
): Edge[] {
  if (!anchor) return [];
  const made: Edge[] = [];
  const liveEdges = [...edges];
  const addEdge = (source: string, target: string, port: string) => {
    if (createsCycle(source, target, liveEdges)) return;
    const edge = { id: `${source}-${target}-${port}`, source, target, sourceHandle: port, targetHandle: port };
    made.push(edge);
    liveEdges.push(edge);
  };

  // image and video are alternatives on dual-port nodes (backend rejects both
  // wired) — once one is connected, never auto-wire the other into the same node
  const alt = (port: string) => (port === "image" ? "video" : port === "video" ? "image" : null);

  // Nodes downstream of the anchor: pulling the new node's inputs from one of
  // these wires backwards around the anchor — and the resulting edge makes the
  // anchor-input fill below a cycle, so it silently never happens (e.g. a
  // Visual Prompt for SigLIP's open prompts port grabbing its image from the
  // Filter downstream of SigLIP, which then blocks the prompts edge itself).
  const downstream = new Set<string>([anchor.id]);
  const stack = [anchor.id];
  while (stack.length) {
    const id = stack.pop()!;
    for (const e of edges)
      if (e.source === id && !downstream.has(e.target)) {
        downstream.add(e.target);
        stack.push(e.target);
      }
  }

  // group boxes have no ports — only flow nodes can be wired
  const candidates = [anchor, ...[...existing].reverse().filter((n) => n.type === "flow" && !downstream.has(n.id))];
  // check ports the anchor can supply first: image is listed before video on
  // dual-port nodes, so on a mixed canvas a stray image node would otherwise
  // win the image/video race and the anchor's video would never get wired
  const anchorOuts = (anchor.data as FlowNodeData).outputs;
  const ports = [...item.inputs].sort((p, q) => Number(anchorOuts.includes(q)) - Number(anchorOuts.includes(p)));
  for (const port of ports) {
    const a = alt(port);
    if (a && made.some((e) => e.target === newNodeId && e.targetHandle === a)) continue;
    const source = candidates.find((n) => (n.data as FlowNodeData).outputs.includes(port));
    if (source) addEdge(source.id, newNodeId, port);
  }

  const anchorData = anchor.data as FlowNodeData;
  const anchorConnected = new Set(edges.filter((e) => e.target === anchor.id).map((e) => e.targetHandle));
  for (const port of anchorData.inputs) {
    if (anchorConnected.has(port) || anchorConnected.has(alt(port) ?? "")) continue;
    if (item.outputs.includes(port)) {
      addEdge(newNodeId, anchor.id, port);
      anchorConnected.add(port);
    }
  }

  return made;
}

// "Auto arrange": layer left-to-right by topological depth (longest path from
// a source), then stack each column vertically, centered on 0 and spaced by
// real node sizes. A group and its children move as one block: child edges
// are lifted to the group for layering, children keep their relative spots.
// Plain Kahn's-algorithm pass — the graphs here are small DAGs, no need for
// a dagre/elk dependency to get a readable layout.
const LAYOUT_GAP_X = 60;
const LAYOUT_GAP_Y = 40;

export function layoutNodes(nodes: Node[], edges: Edge[]): Node[] {
  const parentOf: Record<string, string> = {};
  for (const n of nodes) if (n.parentId) parentOf[n.id] = n.parentId;
  const lift = (id: string) => parentOf[id] ?? id;
  const blocks = nodes.filter((n) => !n.parentId);

  const outgoing: Record<string, string[]> = {};
  const indeg: Record<string, number> = {};
  for (const n of blocks) {
    outgoing[n.id] = [];
    indeg[n.id] = 0;
  }
  for (const e of edges) {
    const s = lift(e.source);
    const t = lift(e.target);
    if (s === t) continue; // edge internal to one group
    outgoing[s]?.push(t);
    indeg[t] = (indeg[t] ?? 0) + 1;
  }

  const depth: Record<string, number> = {};
  const queue = blocks.filter((n) => indeg[n.id] === 0).map((n) => n.id);
  for (const id of queue) depth[id] = 0;
  while (queue.length) {
    const id = queue.shift()!;
    for (const next of outgoing[id]) {
      depth[next] = Math.max(depth[next] ?? 0, depth[id] + 1);
      if (--indeg[next] === 0) queue.push(next);
    }
  }
  for (const n of blocks) depth[n.id] ??= 0; // leftover only if a cycle slipped through

  const columns: Record<number, Node[]> = {};
  for (const n of blocks) (columns[depth[n.id]] ??= []).push(n);

  const pos: Record<string, { x: number; y: number }> = {};
  let x = 0;
  for (const d of Object.keys(columns).map(Number).sort((a, b) => a - b)) {
    const col = columns[d];
    let y = -(col.reduce((s, n) => s + nodeH(n), 0) + (col.length - 1) * LAYOUT_GAP_Y) / 2;
    for (const n of col) {
      pos[n.id] = { x, y };
      y += nodeH(n) + LAYOUT_GAP_Y;
    }
    x += Math.max(...col.map(nodeW)) + LAYOUT_GAP_X;
  }
  // grouped children keep their parent-relative positions untouched
  return nodes.map((n) => (pos[n.id] ? { ...n, position: pos[n.id] } : n));
}

// Save/load: full canvas state as JSON — nodes (incl. group boxes) and edges,
// minus xyflow runtime fields. Per-node config edits (`inputs` overlay) are
// merged into each node's config so the file is self-contained — except image
// payloads: any data: URL (or list of them) is dropped, the saved file holds
// only the graph structure, never uploaded pixels.
const isDataUrl = (v: unknown) => typeof v === "string" && v.startsWith("data:");
// Reset (not delete) scrubbed keys so the config UI still renders the field on
// load. `labels` (Load Image's per-image folder labels, LogReg's per-item
// training labels) is per-item data like the pixels, not graph structure.
// `name` is an upload display hint that only ever lives in the overlay — drop it.
const scrubConfig = (config: Record<string, unknown>) =>
  Object.fromEntries(
    Object.entries(config)
      .filter(([k]) => k !== "name")
      .map(([k, v]) => {
        if (isDataUrl(v) || k === "labels") return [k, Array.isArray(v) ? [] : ""];
        if (Array.isArray(v) && v.some(isDataUrl)) return [k, v.filter((x) => !isDataUrl(x))];
        return [k, v];
      }),
  );

export function serializeFlow(nodes: Node[], edges: Edge[], inputs: Record<string, Record<string, unknown>>): string {
  const cleanNodes = nodes.map((n) => {
    const { measured, selected, dragging, ...rest } = n;
    if (n.type === "flow") {
      const d = rest.data as FlowNodeData;
      rest.data = { ...d, config: scrubConfig({ ...d.config, ...inputs[n.id] }) };
    }
    return rest;
  });
  const cleanEdges = edges.map(({ id, source, target, sourceHandle, targetHandle }) => ({
    id,
    source,
    target,
    sourceHandle: sourceHandle ?? null,
    targetHandle: targetHandle ?? null,
  }));
  return JSON.stringify({ nodes: cleanNodes, edges: cleanEdges }, null, 2);
}

export function deserializeFlow(json: string): { nodes: Node[]; edges: Edge[] } {
  const { nodes, edges } = JSON.parse(json);
  if (!Array.isArray(nodes) || !Array.isArray(edges)) throw new Error("not a workflow file: expected { nodes, edges }");
  // keep future createNode/makeGroup ids clear of the loaded ones
  for (const n of nodes) {
    const m = /-(\d+)$/.exec(n.id ?? "");
    if (m) seq = Math.max(seq, +m[1]);
  }
  return { nodes, edges };
}

let seq = 0;

export function createNode(item: PaletteItem, position: { x: number; y: number }): Node {
  if (item.kind === "app")
    return {
      id: `app-${++seq}`,
      type: "app",
      position,
      data: { title: "App", sections: [] } satisfies AppNodeData,
    };
  return {
    id: `${item.kind}-${++seq}`,
    type: "flow",
    position,
    data: {
      label: item.label,
      kind: item.kind,
      color: item.color,
      inputs: item.inputs,
      outputs: item.outputs,
      required_inputs: item.required_inputs,
      config: { ...item.config },
      config_info: item.config_info,
      config_options: item.config_options,
      config_hf: item.config_hf,
    } satisfies FlowNodeData,
  };
}

// Serializable shape handed to the backend — strips xyflow's runtime fields
// (measured, selected, dragging, ...) and keeps only what a backend needs.
export interface FlowExport {
  nodes: Array<{
    id: string;
    kind: string;
    label: string;
    config: Record<string, unknown>;
    position: { x: number; y: number };
  }>;
  edges: Array<{ id: string; source: string; target: string; sourceHandle: string | null; targetHandle: string | null }>;
}

export function toBackendPayload(nodes: Node[], edges: Edge[]): FlowExport {
  return {
    // group boxes are visual only — the backend rejects nodes without a kind
    nodes: nodes.filter((n) => n.type === "flow").map((n) => {
      const d = n.data as FlowNodeData;
      return { id: n.id, kind: d.kind, label: d.label, config: d.config, position: n.position };
    }),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    })),
  };
}
