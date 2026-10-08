<script lang="ts">
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    Panel,
    ViewportPortal,
    type Node,
    type Edge,
    type OnConnectEnd,
  } from "@xyflow/svelte";
  import "@xyflow/svelte/dist/style.css";
  import { setContext, tick } from "svelte";
  import { scale } from "svelte/transition";

  import AppNode from "./lib/AppNode.svelte";
  import FlowNode from "./lib/FlowNode.svelte";
  import FlowEdge from "./lib/FlowEdge.svelte";
  import GroupNode from "./lib/GroupNode.svelte";
  import Icon from "./lib/Icon.svelte";
  import Toolbar from "./lib/Toolbar.svelte";
  import AppView from "./lib/render/AppView.svelte";
  import {
    API,
    APP_INPUT_KINDS,
    APP_ITEM,
    fetchPalette,
    filesToInput,
    createNode,
    createsCycle,
    nextPosition,
    autoConnectEdges,
    toBackendPayload,
    ancestors,
    absolutePosition,
    nodeW,
    nodeH,
    layoutNodes,
    groupSelected,
    ungroupSelected,
    serializeFlow,
    deserializeFlow,
    FLOW_CTX,
    PORT_COLORS,
    portColor,
    edgeStyle,
    portsMate,
    type AppNodeData,
    type Deployment,
    type NodeResult,
    type PaletteItem,
    type FlowContext,
    type FlowNodeData,
  } from "./lib/flow";
  import { RECIPES, AFFINITY, instantiateRecipe, type Recipe } from "./lib/recipes";
  import ChatPanel from "./lib/ChatPanel.svelte";
  import { insertPlan, type ChatTurn, type Plan } from "./lib/chat";

  const nodeTypes = { flow: FlowNode, group: GroupNode, app: AppNode };
  // Override xyflow's default edge with one that carries an insert-+ button.
  const edgeTypes = { default: FlowEdge };

  let palette = $state<PaletteItem[]>([]);
  let paletteError = $state("");
  let nodes = $state.raw<Node[]>([]);
  let edges = $state.raw<Edge[]>([]);

  // Refresh-proof canvas: the graph mirrors into localStorage on every change
  // and comes back on load. serializeFlow already scrubs uploaded pixel data,
  // so the payload stays small. Save/Load files remain the explicit workflow.
  try {
    const saved = localStorage.getItem("flow");
    if (saved) {
      const flow = deserializeFlow(saved);
      nodes = flow.nodes;
      edges = flow.edges;
    }
  } catch {} // a corrupt autosave must never brick the app — start empty instead
  $effect(() => {
    const json = serializeFlow(nodes, edges, inputs);
    localStorage.setItem("flow", json);
    // Live sync (write half): mirror the canvas into the backend's session
    // slot, debounced. Skipped when the canvas already matches what the server
    // holds (syncedJson) — that's the echo of adopting an external edit.
    clearTimeout(putTimer);
    if (json !== syncedJson) putTimer = setTimeout(() => pushWorkflow(json), 500);
  });

  // --- Live canvas sync (Phase 3): server session, polling, last-write-wins --
  let syncRev = 0; // last rev this client wrote or adopted — our PUT's echo polls back with this rev and is ignored
  let syncedJson = ""; // serializeFlow output the server holds, from our view — suppresses echo PUTs after adoption
  let putTimer: ReturnType<typeof setTimeout> | undefined;
  let toastMsg = $state("");
  let toastTimer: ReturnType<typeof setTimeout> | undefined;
  function toast(msg: string) {
    toastMsg = msg;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toastMsg = ""), 3000);
  }

  // Swap in the server's canvas. Viewport untouched (reassigning nodes never
  // re-fits). Re-serializing right away yields byte-for-byte what the autosave
  // effect is about to compute, so the swap doesn't PUT itself back.
  function adopt(rev: number, workflow: unknown) {
    syncRev = rev;
    if (!workflow) return;
    try {
      const flow = deserializeFlow(JSON.stringify(workflow));
      nodes = flow.nodes;
      edges = flow.edges;
      syncedJson = serializeFlow(nodes, edges, inputs);
    } catch {} // malformed external workflow — keep the current canvas
  }

  async function pushWorkflow(json: string) {
    try {
      const res = await fetch(`${API}/workflow`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: JSON.parse(json), rev: syncRev }),
      });
      if (res.ok) {
        syncRev = (await res.json()).rev;
        syncedJson = json;
      } else if (res.status === 409) {
        // stale rev: someone else wrote meanwhile — last-write-wins, they won
        const { rev, workflow } = await res.json();
        adopt(rev, workflow);
        toast("Canvas updated externally");
      }
    } catch {} // backend down — localStorage above stays the offline path
  }

  // Read half: an external push shows up here within 2 s; our own writes come
  // back with rev == syncRev and fall through. Any *different* rev is state we
  // haven't seen — including a lower one after a backend restart resets the
  // counter (adopting rev 0 + null keeps the canvas and re-syncs the counter).
  setInterval(async () => {
    try {
      const { rev, workflow } = await (await fetch(`${API}/workflow`)).json();
      if (rev !== syncRev) adopt(rev, workflow);
    } catch {} // backend down — silent, same as the write half
  }, 2000);

  // Color every edge like the ports it connects. Stamped here, in one place,
  // so it covers edges from any source — palette auto-connect, a recipe, a
  // loaded file, or the user dragging a connection (which xyflow adds itself).
  $effect(() => {
    if (edges.some((e) => !e.style)) edges = edges.map((e) => (e.style ? e : { ...e, ...edgeStyle(e.sourceHandle) }));
  });

  // Palette comes from the backend — every node class in backend/nodes.py shows
  // up here, plus the one frontend-only entry (the App node). Canvas starts
  // empty: the user picks the first (input) node themselves.
  fetchPalette()
    .then((p) => (palette = [...p, APP_ITEM]))
    .catch((e) => (paletteError = `backend unreachable — is it running on :8000? (${e})`));

  let results = $state<Record<string, string[]>>({});
  let sizes = $state<FlowContext["sizes"]>({});
  let frameIndices = $state<FlowContext["frameIndices"]>({});
  let classifications = $state<FlowContext["classifications"]>({});
  let counts = $state<FlowContext["counts"]>({});
  let downloads = $state<FlowContext["downloads"]>({});
  let points = $state<FlowContext["points"]>({});
  let videos = $state<FlowContext["videos"]>({});
  let legends = $state<FlowContext["legends"]>({});
  let legendMore = $state<FlowContext["legendMore"]>({});
  let volumes = $state<FlowContext["volumes"]>({});
  let errors = $state<Record<string, string>>({});
  let running = $state<Record<string, boolean>>({});
  let inputs = $state<FlowContext["inputs"]>({});
  // >0 while a run's computed subgraph includes this node — drives the
  // pulsing "in progress" outline. Plain (non-reactive) edge-side counter
  // lives alongside it so overlapping runs don't stomp each other's
  // animated-edge state when one finishes before the other.
  let activeNodes = $state<Record<string, number>>({});
  let attention = $state<Record<string, number>>({});
  const activeEdgeCount: Record<string, number> = {};
  // What each in-flight run still holds a +1 on (non-reactive, keyed by run
  // id). A per-target stop releases the dead branch early and shrinks these,
  // so the run's own cleanup at the end doesn't release the same ids twice.
  const runScopes: Record<string, { scope: Set<string>; touched: Edge[] }> = {};
  let progress = $state<FlowContext["progress"]>({});
  // Runs in flight (one per ▶ press — several can overlap): run id -> that
  // run's [nodeId, targetId] pairs. What Stop cancels (all of a run, or one
  // target of it) and what /progress is polled for.
  let activeRuns = $state<Record<string, [string, string][]>>({});
  let stopping = $state(false);
  let anyRunning = $derived(Object.keys(activeRuns).length > 0);

  // The graph as the backend sees it: stored configs with each node's live
  // edits (picked files, typed values) merged on top.
  function backendGraph() {
    const base = toBackendPayload(nodes, edges);
    return {
      ...base,
      nodes: base.nodes.map((n) => {
        const ov = inputs[n.id];
        return ov ? { ...n, config: { ...n.config, ...ov } } : n;
      }),
    };
  }

  // Run the graph up to each [nodeId, targetId] pair in ONE backend pass: the
  // engine's cache is shared across targets, so a SAM3 feeding two viewers runs
  // once. The two ids differ when a node previews its upstream image (Visual
  // Prompt, Crop) — results land under `nodeId`, the graph is cut at `targetId`.
  async function runTargets(pairs: [string, string][]) {
    if (!pairs.length) return;
    const runId = crypto.randomUUID();
    const targets = [...new Set(pairs.map(([, t]) => t))];
    const scope = new Set(targets.flatMap((t) => [...ancestors(t, edges)])); // subgraph the backend computes
    for (const id of scope) errors[id] = ""; // this run re-decides their error state
    for (const [nodeId] of pairs) {
      errors[nodeId] = "";
      running[nodeId] = true;
    }
    for (const id of scope) activeNodes[id] = (activeNodes[id] ?? 0) + 1;
    const touched = edges.filter((e) => scope.has(e.target));
    for (const e of touched) activeEdgeCount[e.id] = (activeEdgeCount[e.id] ?? 0) + 1;
    edges = edges.map((e) => (scope.has(e.target) ? { ...e, animated: true } : e));
    const rs = (runScopes[runId] = { scope, touched });
    activeRuns[runId] = pairs;
    const poll = setInterval(() => pollProgress(runId), 400);
    try {
      const payload = { ...backendGraph(), targets, run_id: runId };
      const res = await fetch(`${API}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      // no `results` at all = the request never reached run_graph (bad payload,
      // crash) — say so with the status instead of a bare "unknown error"
      for (const [nodeId, target] of pairs)
        show(nodeId, json.results?.[target] ?? { error: json.error ?? `backend rejected the run (HTTP ${res.status})` });
    } catch (e) {
      // RangeError = JSON payload past V8's max string length, not a dead backend
      const msg =
        e instanceof RangeError
          ? "payload too large for the browser — use fewer/smaller files"
          : `backend unreachable — is it running on :8000? (${e})`;
      for (const [nodeId] of pairs) errors[nodeId] = msg;
    } finally {
      clearInterval(poll);
      // release through rs/activeRuns, not the locals: a per-target stop may
      // have already released part of the branch and shrunk these to what's
      // still held (and the user may have re-run a stopped node since)
      for (const [nodeId] of activeRuns[runId] ?? []) running[nodeId] = false;
      delete activeRuns[runId];
      if (!anyRunning) stopping = false;
      for (const id of rs.scope) activeNodes[id] = Math.max(0, (activeNodes[id] ?? 1) - 1);
      for (const id of rs.scope) delete progress[id];
      for (const e of rs.touched) activeEdgeCount[e.id] = Math.max(0, (activeEdgeCount[e.id] ?? 1) - 1);
      edges = edges.map((e) => (e.animated && !activeEdgeCount[e.id] ? { ...e, animated: false } : e));
      delete runScopes[runId];
    }
  }

  // A View node's detected objects, thumbs made fetchable. Always assigned
  // (missing -> []) so a re-run that lost its overlays clears the stale chips.
  const shapeLegend = (legend: any[] | undefined) =>
    (legend ?? []).map((e) => ({ ...e, thumb: e.thumb ? API + e.thumb : undefined }));

  // Dispatch one target's result onto the node that displays it.
  function show(nodeId: string, r: Record<string, any>) {
    if (r.cancelled) return; // Stop was pressed — leave the last result standing
    if (r.classification) classifications[nodeId] = r.classification;
    else if (r.counts) counts[nodeId] = r.counts;
    else if (r.download) downloads[nodeId] = { url: API + r.download, summary: r.summary ?? "" };
    else if (r.points) {
      points[nodeId] = r.points;
      // per-point hover thumbnails, one per batch item — same shape results already holds
      if (r.thumbnails) results[nodeId] = r.thumbnails;
    } else if (r.images) {
      // /media URLs, so the browser fetches only the item actually on screen
      results[nodeId] = r.images.map((u: string) => API + u);
      sizes[nodeId] = r.sizes;
      frameIndices[nodeId] = r.frame_indices ?? [];
      legends[nodeId] = shapeLegend(r.legend);
      legendMore[nodeId] = r.legend_more ?? 0;
    } else if (r.videos) {
      videos[nodeId] = r.videos;
      legends[nodeId] = shapeLegend(r.legend);
      legendMore[nodeId] = r.legend_more ?? 0;
    } else if (r.volumes) volumes[nodeId] = r.volumes;
    else {
      errors[nodeId] = r.error ?? "unknown error";
      if (r.node && r.node !== nodeId) {
        const kind = (nodes.find((n) => n.id === r.node)?.data as FlowNodeData | undefined)?.kind;
        // an empty prompt node opens for input instead of showing the error
        if (kind === "text_prompt" || kind === "visual_prompt") attention[r.node] = (attention[r.node] ?? 0) + 1;
        else errors[r.node] = r.error;
      }
    }
  }

  // How far the slow nodes of a run have got. Backend-side each one counts its
  // own work units (frames, images, clips); a failed poll is ignored — the run's
  // own request is what reports real failures.
  async function pollProgress(runId: string) {
    try {
      const json = await (await fetch(`${API}/progress/${runId}`)).json();
      if (!activeRuns[runId]) return; // finished while this poll was in flight — its bars are already cleared
      // gate on the run's current scope: a per-target stop shrinks it, so a
      // stale poll dispatched before the stop can't resurrect dead bars
      const rs = runScopes[runId];
      for (const [id, done] of Object.entries(json.nodes ?? {}))
        if (rs?.scope.has(id)) progress[id] = done as [number, number];
    } catch {}
  }

  const run = (nodeId: string, targetId = nodeId) => runTargets([[nodeId, targetId]]);
  const runAll = () => runTargets(displayIds.map((id) => [id, id] as [string, string]));

  // The backend cancels at the next node boundary or work unit, so the request
  // itself stays open until the model's current chunk is done — hence "Stopping…".
  // With a nodeId (a node's own ■ Stop), only that node's target is cancelled;
  // upstream nodes shared with other running targets keep computing. The dead
  // branch is released right away — its pulse and edge dashes go back to idle —
  // while whatever a live target still needs keeps its running look.
  function stop(nodeId?: string) {
    if (nodeId === undefined) {
      stopping = true;
      for (const id of Object.keys(activeRuns)) fetch(`${API}/stop/${id}`, { method: "POST" });
      return;
    }
    for (const [runId, pairs] of Object.entries(activeRuns)) {
      const kept = pairs.filter(([n]) => n !== nodeId);
      if (kept.length === pairs.length) continue;
      for (const [, target] of pairs.filter(([n]) => n === nodeId))
        fetch(`${API}/stop/${runId}?target=${encodeURIComponent(target)}`, { method: "POST" });
      activeRuns[runId] = kept;
      // release what no remaining target of this run needs; shrinking rs keeps
      // the run's finally block from releasing the same ids a second time
      const rs = runScopes[runId];
      const needed = new Set(kept.flatMap(([, t]) => [...ancestors(t, edges)]));
      for (const id of rs.scope) {
        if (needed.has(id)) continue;
        rs.scope.delete(id);
        activeNodes[id] = Math.max(0, (activeNodes[id] ?? 1) - 1);
        delete progress[id];
      }
      rs.touched = rs.touched.filter((e) => {
        if (needed.has(e.target)) return true;
        activeEdgeCount[e.id] = Math.max(0, (activeEdgeCount[e.id] ?? 1) - 1);
        return false;
      });
      edges = edges.map((e) => (e.animated && !activeEdgeCount[e.id] ? { ...e, animated: false } : e));
    }
    running[nodeId] = false; // its target is dead — the button can offer ▶ Run again
  }

  // Nodes that display something: exactly the ones with a ▶ Run button (see
  // FlowNode) — no output ports, and not a Custom Code node (its ports are typed
  // in, so an empty list means "none picked yet", not "output node"). Wired ones
  // only: a viewer just dropped on the canvas has nothing to compute, and Run
  // All would only paint it with "input 'image' not connected".
  let displayIds = $derived(
    nodes
      .filter(
        (n) =>
          n.type === "flow" &&
          !(n.data as FlowNodeData).outputs.length &&
          (n.data as FlowNodeData).kind !== "custom" &&
          edges.some((e) => e.target === n.id),
      )
      .map((n) => n.id),
  );

  function upstream(nodeId: string, port: string): string | null {
    const e = edges.find((e) => e.target === nodeId && (e.targetHandle ?? "image") === port);
    return e?.source ?? null;
  }

  // --- App node hosting ------------------------------------------------------
  // One App node's page spec, resolved against the live canvas: sections whose
  // node was deleted are dropped ("missing" shows on the App node itself),
  // labels fall back to the referenced node's label, and the display sections
  // double as the page's run targets.
  function appPage(nodeId: string) {
    const nd = nodes.find((n) => n.id === nodeId)?.data as AppNodeData | undefined;
    const byId = new Map(nodes.filter((n) => n.type === "flow").map((n) => [n.id, n.data as FlowNodeData]));
    const sections = (nd?.sections ?? []).flatMap((s) => {
      const node = byId.get(s.node);
      return node
        ? [{ node: s.node, kind: node.kind, label: s.label || node.label, description: s.description || undefined }]
        : [];
    });
    return {
      title: nd?.title || "App",
      sections,
      displayIds: sections.filter((s) => !(s.kind in APP_INPUT_KINDS)).map((s) => s.node),
    };
  }

  // The App node whose page is open (its View button).
  let appViewId = $state<string | null>(null);
  // Where the overlay grows from: the App node's on-screen center at open
  // time, in overlay-local coords (viewport minus the overlay's 20px inset).
  let appOrigin = $state("50% 50%");
  let appOpen = $derived(appViewId ? appPage(appViewId) : null);
  let appRunning = $derived(appOpen?.displayIds.some((id) => running[id]) ?? false);
  // The editor host's per-section results, straight from the canvas stores —
  // the page and the canvas nodes always show the same thing.
  let appResults = $derived.by(() => {
    const out: Record<string, NodeResult> = {};
    for (const id of appOpen?.displayIds ?? [])
      out[id] = {
        images: results[id],
        classification: classifications[id],
        counts: counts[id],
        download: downloads[id],
        points: points[id],
        videos: videos[id]?.map((v) => API + v),
        error: errors[id] || undefined,
      };
    return out;
  });
  // An app-page upload lands in the same config overlay as the Load node's own
  // file button — the canvas node reflects it, and runs pick it up as usual.
  async function appFiles(nodeId: string, media: "image" | "video" | "volume", files: File[]) {
    try {
      inputs[nodeId] = await filesToInput(files, media);
    } catch (err) {
      errors[nodeId] = `upload failed — is the backend running on :8000? (${err})`;
    }
  }

  // Live deployments, synced from the backend on load and after every
  // deploy/stop — the toolbar popover and App-node status read these.
  let deployments = $state<Deployment[]>([]);
  async function syncDeployments() {
    try {
      deployments = await (await fetch(`${API}/deployments`)).json();
    } catch {} // backend down — the deploy/run paths report that already
  }
  syncDeployments();

  async function stopDeployment(name: string) {
    await fetch(`${API}/deployments/${encodeURIComponent(name)}`, { method: "DELETE" });
    syncDeployments();
  }

  setContext<FlowContext>(FLOW_CTX, {
    run,
    runAll,
    stop,
    upstream,
    insertOnEdge,
    canInsert: (src, targetId, tgt) => splices(src, targetId, tgt).length > 0,
    results,
    sizes,
    frameIndices,
    classifications,
    counts,
    downloads,
    points,
    videos,
    legends,
    legendMore,
    volumes,
    errors,
    running,
    inputs,
    activeNodes,
    attention,
    progress,
    openApp: (nodeId) => {
      const r = document.querySelector(`.svelte-flow__node[data-id="${CSS.escape(nodeId)}"]`)?.getBoundingClientRect();
      appOrigin = r ? `${r.x + r.width / 2 - 20}px ${r.y + r.height / 2 - 20}px` : "50% 50%";
      appViewId = nodeId;
    },
    deployApp,
    stopDeployment,
    // getters, not values: plain copies into this object would never update
    // in the nodes reading them
    get deployments() {
      return deployments;
    },
    get stopping() {
      return stopping;
    },
  });

  // Dark mode: manual toggle, defaulting to the OS preference on first visit.
  let dark = $state(
    localStorage.getItem("theme") === "dark" ||
      (!localStorage.getItem("theme") && matchMedia("(prefers-color-scheme: dark)").matches),
  );
  $effect(() => localStorage.setItem("theme", dark ? "dark" : "light"));

  let sidebarOpen = $state(localStorage.getItem("sidebar") !== "closed");
  $effect(() => localStorage.setItem("sidebar", sidebarOpen ? "open" : "closed"));

  // Recipe gallery: fills the empty canvas as the starting screen, and comes
  // back over a busy canvas via the toolbar Tasks button.
  let showTasks = $state(false);

  // MCP hookup dialog: the one-command registration for external coding
  // agents. The backend fills in its checkout path so the command is
  // copy-paste ready; until (or unless) it answers, a placeholder stands in.
  let showMcp = $state(false);
  let mcpCopied = $state(false);
  let mcpDir = $state("<repo>/backend");
  fetch(`${API}/agent-setup`)
    .then(async (r) => (mcpDir = (r.ok && (await r.json()).backend_dir) || mcpDir))
    .catch(() => {}); // backend down/older — the placeholder stands in
  let mcpCmd = $derived(`claude mcp add kumoflow -- uv run --directory ${mcpDir} --extra agent mcp_server.py`);
  function copyMcp() {
    navigator.clipboard.writeText(mcpCmd);
    mcpCopied = true;
    setTimeout(() => (mcpCopied = false), 1500);
  }

  let searchEl = $state<HTMLInputElement | null>(null);

  // One searchable popup serves all three add-at-cursor flows:
  //  quick   — double-click on empty canvas; adds unwired (a spatial add is a
  //            statement about position, not connection)
  //  connect — a connection drag dropped on empty canvas; adds pre-wired to
  //            the dragged handle
  //  insert  — the + button on an edge (FlowEdge); splices the node into it
  type MenuKind =
    | { kind: "quick" }
    | { kind: "connect"; port: string; nodeId: string; fromSource: boolean }
    | { kind: "insert"; edgeId: string; source: string; target: string; src: string; tgt: string };
  let menu = $state<null | (MenuKind & { x: number; y: number; flow: { x: number; y: number } })>(null);
  let menuSearch = $state("");

  function openMenu(kind: MenuKind, clientX: number, clientY: number, flow: { x: number; y: number }) {
    menuSearch = "";
    menu = {
      ...kind,
      x: Math.min(clientX, window.innerWidth - 200),
      y: Math.min(clientY, window.innerHeight - 320),
      flow,
    };
  }

  function onCanvasDblClick(e: MouseEvent) {
    if (!(e.target as Element).classList?.contains("svelte-flow__pane")) return;
    const pane = document.querySelector(".svelte-flow")!.getBoundingClientRect();
    const t = new DOMMatrixReadOnly(getComputedStyle(document.querySelector(".svelte-flow__viewport")!).transform);
    openMenu({ kind: "quick" }, e.clientX, e.clientY, {
      x: (e.clientX - pane.left - t.e) / t.a,
      y: (e.clientY - pane.top - t.f) / t.a,
    });
  }

  // The + button on an edge: capture its endpoints/ports now, so picking a
  // node later can splice without re-finding the edge.
  function insertOnEdge(edgeId: string, mid: { x: number; y: number }, e: MouseEvent) {
    const edge = edges.find((ed) => ed.id === edgeId);
    if (!edge) return;
    openMenu(
      {
        kind: "insert",
        edgeId,
        source: edge.source,
        target: edge.target,
        src: edge.sourceHandle ?? "image",
        tgt: edge.targetHandle ?? "image",
      },
      e.clientX,
      e.clientY,
      mid,
    );
  }

  // ponytail: subsequence scorer, ~10 lines — consecutive-run + word-start
  // bonuses cover "clsfy" → "Classify"; a ranking library would be overkill.
  function fuzzyScore(query: string, text: string): number {
    const q = query.toLowerCase();
    const t = text.toLowerCase();
    let qi = 0;
    let score = 0;
    let run = 0;
    for (let ti = 0; ti < t.length && qi < q.length; ti++) {
      if (t[ti] !== q[qi]) {
        run = 0;
        continue;
      }
      run++;
      score += run + (ti === 0 || !/[a-z0-9]/.test(t[ti - 1]) ? 3 : 0);
      qi++;
    }
    return qi === q.length ? score : -1;
  }

  // Best fuzzy score across a few haystacks, or -1 when none match.
  const fuzzyBest = (q: string, ...texts: string[]) => Math.max(...texts.map((t) => fuzzyScore(q, t)));

  // A node's tooltip + search text: the docstring's first paragraph, whitespace-collapsed.
  const blurb = (it: PaletteItem) => it.doc?.split("\n\n")[0].replace(/\s+/g, " ") ?? "";

  // Recipe-derived "how often does b follow a" count — 0 for any unseen pair.
  const affinity = (a: string, b: string) => AFFINITY[a]?.[b] ?? 0;

  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      menu = null;
      appViewId = null;
      if (started) showTasks = false;
      return;
    }
    const el = e.target as HTMLElement;
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable) return;
    if (e.key === "/" || (e.key.toLowerCase() === "k" && (e.ctrlKey || e.metaKey))) {
      e.preventDefault();
      sidebarOpen = true;
      tick().then(() => searchEl?.focus());
    }
  }

  // fitView is handed up by Toolbar, the nearest component inside SvelteFlow's context
  let fitView: ((opts?: { duration?: number }) => void) | undefined = $state();

  async function arrange() {
    nodes = layoutNodes(nodes, edges);
    await tick(); // let xyflow pick up the new node positions before measuring bounds
    fitView?.({ duration: 200 });
  }

  function save() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([serializeFlow(nodes, edges, inputs)], { type: "application/json" }));
    a.download = "workflow.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // Deploy: snapshot the graph server-side and put it live in one dialog —
  // the backend writes the artifact AND spawns deploy.py on the chosen port.
  // Both entry points share it: the toolbar button (API-only, no ui) and an
  // App node's Deploy (ui attached, targets narrowed to its display sections).
  let deployInfo: {
    form?: boolean;
    app?: string; // App node id when deploying an app page
    name?: string;
    url?: string;
    path?: string;
    command?: string;
    port?: number;
    error?: string;
  } | null = $state(null);
  let deployName = $state("pipeline");
  let deployPort: number | null = $state(8001);
  const showModal = (el: HTMLDialogElement) => el.showModal();

  // Mirror of the backend's name sanitizer — lets the dialog find the stored
  // port of the deployment this name would replace.
  const safeName = (name: string) => name.replace(/[^\w.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || "pipeline";

  // Replacing an existing name keeps its port (shared URLs stay stable);
  // a new name gets the first port no live deployment holds.
  function suggestPort(name: string): number {
    const existing = deployments.find((d) => d.name === safeName(name));
    if (existing) return existing.port;
    const used = new Set(deployments.filter((d) => d.alive).map((d) => d.port));
    let p = 8001;
    while (used.has(p)) p++;
    return p;
  }

  function openDeployForm(name: string, app?: string) {
    deployName = name;
    deployPort = suggestPort(name);
    deployInfo = { form: true, app };
  }

  function deploy() {
    if (!displayIds.length) {
      deployInfo = { error: "Wire up an output node (e.g. View Image) first — the deployed pipeline runs what Run All runs." };
      return;
    }
    openDeployForm(deployName);
  }

  // An App node's Deploy button: name pre-filled from the app title, targets
  // narrowed to its display sections.
  function deployApp(nodeId: string) {
    const page = appPage(nodeId);
    if (!page.displayIds.length) {
      deployInfo = { error: "Add at least one result section (View Image, Count, …) to the App node first — the page runs what its sections show." };
      return;
    }
    openDeployForm(page.title, nodeId);
  }

  async function deployNow() {
    const app = deployInfo?.app;
    try {
      const body: Record<string, unknown> = { name: deployName || "pipeline", port: deployPort || 8001 };
      if (app) {
        const page = appPage(app);
        body.graph = { ...backendGraph(), targets: page.displayIds };
        // the ui field is the artifact's page spec: section labels baked here,
        // so the deployed page needs no access to canvas node labels
        body.ui = { title: page.title, sections: page.sections.map(({ kind, ...s }) => s) };
      } else {
        body.graph = { ...backendGraph(), targets: displayIds };
      }
      const res = await fetch(`${API}/deploy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      if (!res.ok) {
        let msg = text;
        try {
          msg = JSON.parse(text).detail ?? text; // FastAPI wraps HTTPException in {"detail"}
        } catch {}
        throw new Error(msg);
      }
      const d = JSON.parse(text);
      // remember which deployment is this App node's, for its live/Stop state
      if (app) nodes = nodes.map((n) => (n.id === app ? { ...n, data: { ...n.data, deployment: d.name } } : n));
      deployInfo = d;
      syncDeployments();
    } catch (e) {
      deployInfo = { error: `Deploy failed — ${e instanceof Error ? e.message : e}` };
    }
  }

  async function load(file: File) {
    try {
      const flow = deserializeFlow(await file.text());
      // saved configs already include the overlay edits; a stale overlay from
      // the previous canvas would shadow them (mutate, not reassign — the
      // context object shared with FlowNode holds a reference to `inputs`)
      for (const k of Object.keys(inputs)) delete inputs[k];
      nodes = flow.nodes;
      edges = flow.edges;
    } catch (e) {
      window.alert(`Could not load workflow: ${e}`);
    }
  }

  // Bundle the currently selected nodes into a labeled subflow box (Svelte
  // Flow "group" node). window.prompt is enough UI for naming it; cancel = no-op.
  function group() {
    const label = window.prompt("Group name", "Group");
    if (label === null) return;
    nodes = groupSelected(nodes, label || "Group");
  }

  // Contextual actions, shown bottom-center only while applicable.
  let canGroup = $derived(nodes.filter((n) => n.selected && n.type === "flow" && !n.parentId).length >= 2);
  let canUngroup = $derived(nodes.some((n) => n.selected && n.type === "group"));

  // The node to place a new addition next to: the selected node, else the
  // last one added, else undefined (empty canvas -> the original cascade).
  // Group boxes don't qualify — they have no ports to auto-wire against.
  function anchorNode(): Node | undefined {
    const flow = nodes.filter((n) => n.type === "flow");
    return flow.find((n) => n.selected) ?? flow[flow.length - 1];
  }

  function add(item: PaletteItem) {
    const anchor = anchorNode();
    const node = createNode(item, nextPosition(anchor, nodes));
    const newEdges = autoConnectEdges(item, node.id, anchor, nodes, edges);
    nodes = [...nodes, node];
    if (newEdges.length) edges = [...edges, ...newEdges];
  }

  function addRecipe(recipe: Recipe) {
    const origin = nextPosition(anchorNode(), nodes);
    const { nodes: newNodes, edges: newEdges } = instantiateRecipe(recipe, byKind, origin);
    nodes = [...nodes, ...newNodes];
    edges = [...edges, ...newEdges];
  }

  // The chat card's "Add to canvas": a validated plan lands like a recipe —
  // laid out, wired, at the same origin addRecipe would pick. From here on
  // it's a normal subgraph (autosave, undo, run).
  let chatPanel = $state<{ show: () => void } | null>(null);
  function insertFromChat(plan: Plan, outputHandling: ChatTurn["output_handling"]) {
    const origin = nextPosition(anchorNode(), nodes);
    const sub = insertPlan(plan, byKind, origin, outputHandling === "app");
    nodes = [...nodes, ...sub.nodes];
    edges = [...edges, ...sub.edges];
  }

  const onconnectend: OnConnectEnd = (event, state) => {
    // toHandle set = dropped on (or snapped to) a real handle — normal connect
    if (!state.fromHandle || state.toHandle || !state.to) return;
    const { clientX, clientY } = "changedTouches" in event ? event.changedTouches[0] : event;
    // state.to is container-relative pixels (despite the docs), not flow coords —
    // undo the pan/zoom transform xyflow puts on its viewport pane.
    const t = new DOMMatrixReadOnly(getComputedStyle(document.querySelector(".svelte-flow__viewport")!).transform);
    openMenu(
      {
        kind: "connect",
        port: state.fromHandle.id ?? "image",
        nodeId: state.fromHandle.nodeId,
        fromSource: state.fromHandle.type === "source",
      },
      clientX,
      clientY,
      { x: (state.to.x - t.e) / t.a, y: (state.to.y - t.f) / t.a },
    );
  };

  // Ports of an edge's target a spliced node could feed: the wire's own port
  // first, then any still-unwired inputs. image/video are alternatives on
  // dual-port nodes (backend rejects both wired), so a wired one also blocks
  // its counterpart.
  const IV_ALT: Record<string, string> = { image: "video", video: "image" };
  function openTargetPorts(targetId: string, tgt: string): string[] {
    const data = nodes.find((n) => n.id === targetId)?.data as FlowNodeData | undefined;
    const blocked = new Set([tgt]);
    for (const e of edges) if (e.target === targetId) blocked.add(e.targetHandle ?? "image");
    for (const b of [...blocked]) if (IV_ALT[b]) blocked.add(IV_ALT[b]);
    return [tgt, ...(data?.inputs ?? []).filter((p) => !blocked.has(p))];
  }

  // Palette nodes that can be inserted on an edge carrying src into (targetId,
  // tgt): they accept what flows through it AND produce something the target
  // node can take — on the wire's port (splice) or any open input (tap, e.g.
  // sam3 feeding view's optional masks while the image edge stays). Also
  // drives FlowEdge's + button visibility via canInsert in the context.
  const splices = (src: string, targetId: string, tgt: string) => {
    const open = openTargetPorts(targetId, tgt);
    return palette.filter(
      (it) =>
        relevant(it) &&
        it.inputs.some((i) => portsMate(src, i)) &&
        it.outputs.some((o) => open.some((p) => portsMate(o, p))),
    );
  };

  // Recipe-common candidates first (stable sort: ties keep palette order).
  // No anchor (quick-add) → leave the pool as found. `into` flips the wire
  // direction: candidates feeding INTO the anchor (drag from an input handle).
  const sortByAffinity = (anchor: string | undefined, cands: PaletteItem[], into = false) =>
    anchor
      ? [...cands].sort((a, b) => (into ? affinity(b.kind, anchor) - affinity(a.kind, anchor) : affinity(anchor, b.kind) - affinity(anchor, a.kind)))
      : cands;

  // What the open menu offers: quick = the whole search pool, connect = ports
  // mate with the dragged handle, insert = the splice candidates.
  let menuPool = $derived.by(() => {
    const m = menu;
    if (!m) return [];
    if (m.kind === "quick") return searchPool;
    if (m.kind === "connect") {
      const anchor = (nodes.find((n) => n.id === m.nodeId)?.data as FlowNodeData | undefined)?.kind;
      return sortByAffinity(anchor, palette.filter((it) => relevant(it) && (m.fromSource ? it.inputs : it.outputs).includes(m.port)), !m.fromSource);
    }
    const anchor = (nodes.find((n) => n.id === m.source)?.data as FlowNodeData | undefined)?.kind;
    return sortByAffinity(anchor, splices(m.src, m.target, m.tgt));
  });
  let menuCandidates = $derived(
    menuSearch.trim() ? rank(menuPool, menuSearch.trim(), (it) => [it.label, it.category, blurb(it)]) : menuPool,
  );
  let menuPort = $derived(menu?.kind === "connect" ? menu.port : menu?.kind === "insert" ? menu.src : null);

  function pickMenuItem(item: PaletteItem) {
    const m = menu!;
    // insert: roughly center the node on the edge midpoint instead of hanging
    // it off the cursor — it's replacing a point on the wire, not a drop spot
    const node = createNode(item, m.kind === "insert" ? { x: m.flow.x - 110, y: m.flow.y - 40 } : m.flow);
    nodes = [...nodes, node];
    if (m.kind === "connect") {
      // a brand-new node can't close a cycle, so no createsCycle check needed
      const [source, target] = m.fromSource ? [m.nodeId, node.id] : [node.id, m.nodeId];
      edges = [...edges, { id: `${source}-${target}-${m.port}`, source, target, sourceHandle: m.port, targetHandle: m.port }];
    } else if (m.kind === "insert") {
      const inPort = item.inputs.find((i) => portsMate(m.src, i))!;
      const made = [
        { id: `${m.source}-${node.id}-${inPort}`, source: m.source, target: node.id, sourceHandle: m.src, targetHandle: inPort },
      ];
      const spliceOut = item.outputs.find((o) => portsMate(o, m.tgt));
      if (spliceOut) {
        // splice: the node re-produces the wire's type and replaces it
        edges = edges.filter((e) => e.id !== m.edgeId);
        made.push({ id: `${node.id}-${m.target}-${m.tgt}`, source: node.id, target: m.target, sourceHandle: spliceOut, targetHandle: m.tgt });
      } else {
        // tap: the wire stays; the node enriches the target through every open
        // port it can supply (sam3 on load→view adds boxes AND masks)
        for (const p of openTargetPorts(m.target, m.tgt)) {
          const o = item.outputs.find((o2) => portsMate(o2, p));
          if (o) made.push({ id: `${node.id}-${m.target}-${p}`, source: node.id, target: m.target, sourceHandle: o, targetHandle: p });
        }
      }
      edges = [...edges, ...made];
    }
    menu = null;
  }

  let byKind = $derived.by(() => Object.fromEntries(palette.map((i) => [i.kind, i])));
  let started = $derived(nodes.length > 0);

  // Modalities produced by the input node(s) on the canvas ("image", "video").
  // Processing nodes are only offered when they support one of them — picking
  // the input node decides which downstream nodes are choosable. Input nodes
  // stay offered (so a second source can be added); with no input on the
  // canvas nothing is filtered.
  let canvasModalities = $derived.by(() => {
    const m = new Set<string>();
    for (const n of nodes) {
      if (n.type !== "flow") continue;
      const item = byKind[(n.data as FlowNodeData).kind];
      if (item?.category === "Input") for (const x of item.modalities) m.add(x);
    }
    return m;
  });
  const relevant = (it: PaletteItem) =>
    it.category === "Input" || !canvasModalities.size || it.modalities.some((m) => canvasModalities.has(m));

  // Group the palette by node category, ordered along the dataflow (Input → Output);
  // unknown categories sort to the end. Gated to Input-only until the canvas
  // has its first node — the user picks a starting point before anything else.
  const CAT_ORDER = ["Input", "Transform", "Prompt", "Detect", "Segment", "Track", "Embed", "Classify", "Analyze", "Output", "Custom"];
  let groups = $derived.by(() => {
    const byCat: Record<string, PaletteItem[]> = {};
    for (const it of palette) if (relevant(it)) (byCat[it.category] ??= []).push(it);
    const rank = (c: string) => (CAT_ORDER.indexOf(c) + 1 || 99);
    const entries = Object.entries(byCat).sort(([a], [b]) => rank(a) - rank(b));
    return started ? entries : entries.filter(([cat]) => cat === "Input");
  });

  // The pool every search draws from: Input-only until the canvas has its
  // first node, then whatever matches the canvas modalities.
  let searchPool = $derived((started ? palette : palette.filter((it) => it.category === "Input")).filter(relevant));

  // Suggested next nodes, shown as greyed-out ghost cards with accept/decline
  // and a dashed arrow from the port they'd wire to. Two sources: a leaf's
  // outputs suggest consumers (to the right), a node's open input ports suggest
  // producers (to the left). ≤2 palette nodes qualify → show them; more and
  // recipe affinity picks the top 2 (see topAffinity), none → too ambiguous.
  // Declines are session-only, keyed per (node, kind).
  let suggest = $state(localStorage.getItem("suggest") !== "off");
  $effect(() => localStorage.setItem("suggest", suggest ? "on" : "off"));
  let declined = $state<Record<string, boolean>>({});

  // Fixed card size so placement and arrow endpoints are computable.
  const GHOST_W = 170;
  const GHOST_H = 32;
  const GHOST_GAP = 60;
  const GHOST_ROW = GHOST_H + 12;

  // Where a node's ghost column starts: beside the node, bumped down past any
  // real node the first card would cover (nextPosition's trick) — input ghosts
  // would otherwise land on the upstream node feeding the wired ports.
  function ghostBase(n: Node, side: "in" | "out"): { x: number; y: number } {
    const p = absolutePosition(n, nodes);
    const pos = { x: side === "out" ? p.x + nodeW(n) + GHOST_GAP : p.x - GHOST_GAP - GHOST_W, y: p.y };
    const hit = (m: Node) => {
      if (m.type !== "flow") return false;
      const q = absolutePosition(m, nodes);
      return pos.x < q.x + nodeW(m) && q.x < pos.x + GHOST_W && pos.y < q.y + nodeH(m) && q.y < pos.y + GHOST_H;
    };
    while (nodes.some(hit)) pos.y += GHOST_ROW;
    return pos;
  }

  interface Suggestion {
    anchor: Node;
    item: PaletteItem;
    port: string; // the anchor's port the ghost would wire to
    side: "in" | "out";
    x: number;
    y: number; // card top-left, flow coords
    from: { x: number; y: number }; // arrow start: the port's handle position
  }

  // ≤2 candidates: today's behavior, unchanged. More than that is too
  // ambiguous on its own, so recipe affinity breaks the tie — keep the top 2
  // that actually co-occur with the anchor; none score → no suggestion (same
  // "too ambiguous" outcome as before).
  const topAffinity = (cands: PaletteItem[], score: (it: PaletteItem) => number) =>
    cands.length <= 2
      ? cands
      : cands
          .map((it) => [it, score(it)] as const)
          .filter(([, s]) => s > 0)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 2)
          .map(([it]) => it);

  let suggestions = $derived.by(() => {
    if (!suggest) return [];
    const out: Suggestion[] = [];
    const hasOut = new Set(edges.map((e) => e.source));
    const wired = new Set(edges.map((e) => `${e.target}:${e.targetHandle ?? "image"}`));
    // Handle position in flow coords — FlowNode stacks a node's handles at
    // (i+1)/(n+1) of its height; keep in sync with its `stack` helper.
    const portAt = (n: Node, ports: string[], port: string, right: boolean) => {
      const p = absolutePosition(n, nodes);
      return { x: p.x + (right ? nodeW(n) : 0), y: p.y + (nodeH(n) * (ports.indexOf(port) + 1)) / (ports.length + 1) };
    };
    for (const n of nodes) {
      if (n.type !== "flow") continue;
      const data = n.data as FlowNodeData;
      const ok = (it: PaletteItem) => relevant(it) && !declined[`${n.id}:${it.kind}`];

      // leaf outputs (pooled): what could consume this node's results?
      if (!hasOut.has(n.id) && data.outputs.length) {
        const raw = palette.filter((it) => ok(it) && data.outputs.some((o) => it.inputs.some((i) => portsMate(o, i))));
        const cands = topAffinity(raw, (it) => affinity(data.kind, it.kind));
        if (cands.length) {
          const base = ghostBase(n, "out");
          cands.forEach((item, row) => {
            const port = data.outputs.find((o) => item.inputs.some((i) => portsMate(o, i)))!;
            out.push({ anchor: n, item, port, side: "out", x: base.x, y: base.y + row * GHOST_ROW, from: portAt(n, data.outputs, port, true) });
          });
        }
      }

      // each open input port: what could feed it? (a wired image/video also
      // blocks its dual-port counterpart, same rule as auto-connect)
      let row = 0;
      let base: { x: number; y: number } | null = null;
      for (const port of data.inputs) {
        if (wired.has(`${n.id}:${port}`) || wired.has(`${n.id}:${IV_ALT[port] ?? port}`)) continue;
        const raw = palette.filter((it) => ok(it) && it.outputs.some((o) => portsMate(o, port)));
        const cands = topAffinity(raw, (it) => affinity(it.kind, data.kind));
        if (!cands.length) continue;
        base ??= ghostBase(n, "in");
        for (const item of cands)
          out.push({ anchor: n, item, port, side: "in", x: base.x, y: base.y + row++ * GHOST_ROW, from: portAt(n, data.inputs, port, false) });
      }
    }
    return out;
  });

  function acceptSuggestion(s: Suggestion) {
    const node = createNode(s.item, { x: s.x, y: s.y });
    const newEdges = autoConnectEdges(s.item, node.id, s.anchor, nodes, edges);
    nodes = [...nodes, node];
    if (newEdges.length) edges = [...edges, ...newEdges];
  }

  // Rank by fuzzy score so "clsfy" finds Classify and better matches sort first.
  function rank<T>(list: T[], q: string, texts: (x: T) => string[]): T[] {
    return list
      .map((x) => [x, fuzzyBest(q, ...texts(x))] as const)
      .filter(([, s]) => s >= 0)
      .sort((a, b) => b[1] - a[1])
      .map(([x]) => x);
  }

  // Search bypasses categories entirely and matches recipes + raw nodes flat.
  let search = $state("");
  let matched = $derived.by(() => {
    const q = search.trim();
    if (!q) return null;
    const items = rank(searchPool, q, (it) => [it.label, it.category, blurb(it)]);
    // Recipes stay offered even on an empty canvas: instantiating one always
    // creates its own input node, so picking a recipe *is* a valid first move.
    const recipes = rank(RECIPES, q, (r) => [r.name, r.description]);
    return { items, recipes };
  });
</script>

<div class="layout" class:dark>
  {#if sidebarOpen}
  <aside class="palette">
    <div class="palette__head">
      <h2>Nodes</h2>
      <div class="palette__head-actions">
        <button class="icon-btn" onclick={() => (dark = !dark)} title="Toggle dark mode" aria-label="Toggle dark mode">
          <Icon name={dark ? "sun" : "moon"} size={14} />
        </button>
        <button class="icon-btn" onclick={() => (sidebarOpen = false)} title="Hide node palette" aria-label="Hide node palette">
          <Icon name="chevrons-left" size={14} />
        </button>
      </div>
    </div>
    {#if paletteError}<p class="err">{paletteError}</p>{/if}

    <input
      class="palette__search"
      type="text"
      placeholder="Search nodes…  /"
      bind:value={search}
      bind:this={searchEl}
    />

    {#if matched}
      {#each matched.recipes as recipe}
        <button class="palette__item palette__item--recipe" title={recipe.description} onclick={() => addRecipe(recipe)}>
          {recipe.name}
        </button>
      {/each}
      {#each matched.items as item}
        <button class="palette__item" style:border-left-color={item.color} title={blurb(item)} onclick={() => add(item)}>
          {item.label}
        </button>
      {/each}
      {#if !matched.recipes.length && !matched.items.length}<p class="hint">No matches.</p>{/if}
    {:else}
      {#each groups as [category, items]}
        <details class="palette__cat-group" open>
          <summary class="palette__cat">{category}</summary>
          {#each items as item}
            <button class="palette__item" style:border-left-color={item.color} title={blurb(item)} onclick={() => add(item)}>
              {item.label}
            </button>
          {/each}
        </details>
      {/each}
    {/if}

    <details class="palette__help">
      <summary class="palette__cat">Legend &amp; tips</summary>
      <div class="legend">
        <div class="legend__row"><span class="dot filled"></span>required in / out</div>
        <div class="legend__row"><span class="dot"></span>optional input</div>
        <div class="legend__types">
          {#each Object.entries(PORT_COLORS) as [name, color]}
            <span class="legend__type"><span class="dot filled" style:background={color} style:border-color={color}></span>{name}</span>
          {/each}
        </div>
        <label class="legend__row legend__toggle">
          <input type="checkbox" bind:checked={suggest} />
          Suggest next nodes
        </label>
      </div>
      <p class="hint">
        Double-click the canvas to add a node at the cursor; drag a connection into empty space to add a pre-wired one;
        click the + on a wire to insert a node between two connected ones.
        Select nodes/edges (Shift-drag or Ctrl/Cmd-click for multiple) and press Backspace to delete.
        Selecting 2+ nodes offers Group; selecting a group box offers Ungroup.
      </p>
    </details>
  </aside>
  {:else}
    <button class="palette-opener" onclick={() => (sidebarOpen = true)} title="Show node palette" aria-label="Show node palette">
      <Icon name="chevrons-right" size={14} />
    </button>
  {/if}

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="canvas" ondblclick={onCanvasDblClick}>
    {#if !started || showTasks}
      {#if started}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <div class="tasks-backdrop" onclick={() => (showTasks = false)}></div>
      {/if}
      <div class="tasks" class:tasks--overlay={started}>
        <h3 class="tasks__title">{started ? "Common tasks" : "Start with a common task"}</h3>
        <div class="tasks__grid">
          {#each RECIPES as recipe}
            <button
              class="tasks__card"
              disabled={!palette.length}
              onclick={() => {
                addRecipe(recipe);
                showTasks = false;
              }}
            >
              <span class="tasks__name"><Icon name={recipe.icon} size={15} /> {recipe.name}</span>
              <span class="tasks__desc">{recipe.description}</span>
            </button>
          {/each}
        </div>
        {#if !started}
          <p class="tasks__alt">…or build from scratch: add an input node from the palette</p>
          <button class="tasks__chat" onclick={() => chatPanel?.show()}>
            <Icon name="chat" size={13} /> …or describe what you want to build
          </button>
          <button class="tasks__chat" onclick={() => (showMcp = true)}>
            <Icon name="cpu" size={13} /> …or connect your own coding agent (MCP)
          </button>
        {/if}
      </div>
    {/if}
    <SvelteFlow
      bind:nodes
      bind:edges
      {nodeTypes}
      {edgeTypes}
      fitView
      colorMode={dark ? "dark" : "light"}
      deleteKey={["Backspace", "Delete"]}
      zoomOnDoubleClick={false}
      isValidConnection={(c) => portsMate(c.sourceHandle, c.targetHandle) && !createsCycle(c.source, c.target, edges)}
      {onconnectend}
      proOptions={{ hideAttribution: true }}
    >
      <Controls showLock={false} />
      <Background />
      <MiniMap />
      {#if suggestions.length}
        <ViewportPortal target="front">
          <svg class="suggestion-wires">
            <defs>
              <marker id="suggest-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" />
              </marker>
            </defs>
            {#each suggestions as s}
              {@const ex = s.side === "out" ? s.x - 3 : s.x + GHOST_W + 3}
              {@const ey = s.y + GHOST_H / 2}
              {@const o = s.side === "out" ? 40 : -40}
              <path d="M {s.from.x} {s.from.y} C {s.from.x + o} {s.from.y}, {ex - o} {ey}, {ex} {ey}" marker-end="url(#suggest-arrow)" />
            {/each}
          </svg>
          {#each suggestions as s (`${s.side}:${s.anchor.id}:${s.port}:${s.item.kind}`)}
            <div class="suggestion" style:transform="translate({s.x}px, {s.y}px)">
              <span class="suggestion__label" style:border-left-color={s.item.color} title={blurb(s.item)}>{s.item.label}</span>
              <button class="suggestion__btn suggestion__btn--ok" onclick={() => acceptSuggestion(s)} title="Add {s.item.label}">✓</button>
              <button class="suggestion__btn" onclick={() => (declined[`${s.anchor.id}:${s.item.kind}`] = true)} title="Dismiss suggestion">✕</button>
            </div>
          {/each}
        </ViewportPortal>
      {/if}
      <Panel position="top-right">
        <Toolbar onSave={save} onLoad={load} onTasks={() => (showTasks = !showTasks)} {deployments} onStopDeployment={stopDeployment} bind:fitView />
      </Panel>
      {#if displayIds.length || canGroup || canUngroup}
        <Panel position="bottom-center">
          {#if displayIds.length}
            <button class="action-btn" onclick={arrange} title="Lay out nodes left-to-right by dataflow, then fit the view">
              <Icon name="arrange" size={11} /> Arrange
            </button>
            <button
              class="action-btn action-btn--run"
              onclick={runAll}
              disabled={anyRunning}
              title="Run every display node in one pass — shared work is computed once"
            >
              {#if anyRunning}Running…{:else}<Icon name="play" size={11} /> Run All{/if}
            </button>
            <button class="action-btn" onclick={deploy} title="Save the workflow server-side and get a command to serve it as a standalone API">
              <Icon name="deploy" size={11} /> Deploy
            </button>
          {/if}
          {#if anyRunning}
            <button class="action-btn" onclick={() => stop()}>
              {#if stopping}Stopping…{:else}<Icon name="stop" size={11} /> Stop{/if}
            </button>
          {/if}
          {#if canGroup}
            <button class="action-btn" onclick={group} title="Group the selected nodes into a labeled box">
              <Icon name="group" size={11} /> Group
            </button>
          {/if}
          {#if canUngroup}
            <button class="action-btn" onclick={() => (nodes = ungroupSelected(nodes))}>
              <Icon name="ungroup" size={11} /> Ungroup
            </button>
          {/if}
        </Panel>
      {/if}
    </SvelteFlow>
  </div>

  <ChatPanel bind:this={chatPanel} {byKind} getGraph={() => toBackendPayload(nodes, edges)} onInsert={insertFromChat} />

  {#if menu}
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <div class="connect-menu__backdrop" onclick={() => (menu = null)}></div>
    <div class="connect-menu" style:left="{menu.x}px" style:top="{menu.y}px">
      {#if menuPort}
        <div class="connect-menu__title">
          <span class="dot filled" style:background={portColor(menuPort)} style:border-color={portColor(menuPort)}></span>
          {menuPort}
        </div>
      {/if}
      <!-- svelte-ignore a11y_autofocus -->
      <input
        class="palette__search connect-menu__search"
        type="text"
        placeholder="Add node…"
        autofocus
        bind:value={menuSearch}
        onkeydown={(e) => e.key === "Enter" && menuCandidates.length && pickMenuItem(menuCandidates[0])}
      />
      {#each menuCandidates as item}
        <button class="palette__item" style:border-left-color={item.color} title={blurb(item)} onclick={() => pickMenuItem(item)}>
          {item.label}
        </button>
      {/each}
      {#if !menuCandidates.length}<p class="hint">No matches.</p>{/if}
    </div>
  {/if}

  {#if deployInfo}
    <dialog class="deploy-dialog" use:showModal onclose={() => (deployInfo = null)}>
      {#if deployInfo.form}
        <h3>Deploy</h3>
        <form onsubmit={(e) => (e.preventDefault(), deployNow())}>
          <p>Deployment name</p>
          <!-- svelte-ignore a11y_autofocus -->
          <input class="palette__search" type="text" autofocus bind:value={deployName} onfocus={(e) => e.currentTarget.select()} />
          <p>Port</p>
          <input class="palette__search" type="number" min="1" max="65535" placeholder="8001" bind:value={deployPort} />
          <p class="dialog-hint">Deploying an existing name replaces it in place — its URL stays stable.</p>
          <button class="action-btn" type="submit">Deploy</button>
          <button class="action-btn" type="button" onclick={() => (deployInfo = null)}>Cancel</button>
        </form>
      {:else if deployInfo.error}
        <h3>Deploy</h3>
        <p>{deployInfo.error}</p>
      {:else}
        <h3>Deployed</h3>
        <p>Live at <a href={deployInfo.url} target="_blank">{deployInfo.url}</a></p>
        <p>Artifact: <code>{deployInfo.path}</code> — to serve it yourself (e.g. after a backend restart, from <code>backend/</code>):</p>
        <pre>{deployInfo.command}</pre>
        <p>The same URL is a scriptable API — <code>POST {deployInfo.url}/run</code>, or from Python (<code>backend/client.py</code>):</p>
        <pre>{`from client import Pipeline\nPipeline("localhost:${deployInfo.port}").run(images={"image": "cat.jpg"})`}</pre>
      {/if}
      {#if !deployInfo.form}
        <button class="action-btn" onclick={() => (deployInfo = null)}>Close</button>
      {/if}
    </dialog>
  {/if}

  {#if showMcp}
    <dialog class="deploy-dialog" use:showModal onclose={() => (showMcp = false)}>
      <h3>Connect your coding agent</h3>
      <p>Agent-built workflows show up live on this canvas. For Claude Code:</p>
      <pre>{mcpCmd}</pre>
      <p>Other MCP clients: same command and args. Details in <code>AGENTS.md</code>.</p>
      <button class="action-btn" onclick={copyMcp}>{mcpCopied ? "Copied ✓" : "Copy command"}</button>
      <button class="action-btn" onclick={() => (showMcp = false)}>Close</button>
    </dialog>
  {/if}

  {#if appViewId && appOpen}
    <!-- the App node's View button: the app page over the live editor graph,
         grown out of the node — unsaved edits included, per-node progress for free -->
    <div class="app-overlay" style:transform-origin={appOrigin} transition:scale={{ duration: 220, start: 0.1 }}>
      <button class="app-overlay__close icon-btn" onclick={() => (appViewId = null)} title="Back to the canvas" aria-label="Close app view">
        <Icon name="x" size={16} />
      </button>
      <AppView
        title={appOpen.title}
        sections={appOpen.sections}
        results={appResults}
        running={appRunning}
        canRun={appOpen.displayIds.length > 0}
        {progress}
        onRun={() => runTargets(appOpen!.displayIds.map((id) => [id, id]))}
        onFiles={appFiles}
      />
    </div>
  {/if}

  {#if toastMsg}
    <div class="toast" transition:scale={{ duration: 150, start: 0.9 }}>{toastMsg}</div>
  {/if}
</div>

<svelte:window onkeydown={onWindowKeydown} />

<style>
  .layout {
    display: flex;
    height: 100vh;
    background: var(--bg);
    color: var(--text);
  }
  .toast {
    position: fixed;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 30;
    padding: 8px 14px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    font-size: 13px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .palette {
    width: 200px;
    flex: 0 0 200px;
    padding: 12px;
    border-right: 1px solid var(--border);
    background: var(--sidebar-bg);
    box-sizing: border-box;
    overflow-y: auto;
  }
  .palette__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .palette h2 {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-strong);
    margin: 0;
  }
  .palette__head-actions {
    display: flex;
    gap: 4px;
  }
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    line-height: 1;
    padding: 5px;
  }
  .icon-btn:hover {
    background: var(--hover-bg);
    color: var(--text);
  }
  .palette-opener {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    padding: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .palette-opener:hover {
    background: var(--hover-bg);
    color: var(--text);
  }
  .palette__cat {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 14px 0 6px;
  }
  .palette__cat:first-of-type {
    margin-top: 0;
  }
  .palette__item {
    display: block;
    width: 100%;
    text-align: left;
    padding: 8px 12px;
    margin-bottom: 6px;
    border: 1px solid var(--border-strong);
    border-left: 4px solid #999;
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 13px;
  }
  .palette__item:hover {
    background: var(--hover-bg);
  }
  .palette__item--recipe {
    border-left-color: var(--accent);
    font-weight: 600;
  }
  .palette__search {
    width: 100%;
    box-sizing: border-box;
    padding: 6px 8px;
    margin-bottom: 12px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--input-bg);
    color: var(--text);
    font-size: 13px;
  }
  .palette__cat-group {
    margin-bottom: 4px;
  }
  .palette__cat-group summary.palette__cat {
    margin: 10px 0 6px;
    cursor: pointer;
  }
  .palette__cat-group summary.palette__cat::-webkit-details-marker {
    color: #bbb;
  }
  .palette__help {
    margin-top: 18px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .palette__help summary {
    cursor: pointer;
  }
  .legend {
    font-size: 11px;
    color: var(--muted-strong);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .legend__row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend__types {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    margin-top: 2px;
  }
  .legend__type {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--card-bg);
    border: 2px solid #64748b;
    box-sizing: border-box;
    flex: 0 0 auto;
  }
  .dot.filled {
    background: #64748b;
  }
  .err {
    color: var(--danger);
    font-size: 12px;
  }
  .hint {
    color: var(--muted);
    font-size: 11px;
    margin-top: 12px;
  }
  .canvas {
    position: relative;
    flex: 1;
    min-width: 0;
  }
  .action-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    padding: 7px 14px;
    margin: 0 3px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .action-btn:hover {
    background: var(--hover-bg);
  }
  .action-btn--run {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .action-btn--run:hover {
    background: var(--accent);
  }
  .action-btn:disabled {
    opacity: 0.6;
    cursor: default;
  }
  /* Ghost card for a suggested next node — greyed out until hovered; lives in
     the ViewportPortal, whose parent has pointer-events:none, hence `all`. */
  .suggestion {
    position: absolute;
    pointer-events: all;
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 7px;
    /* fixed size = GHOST_W/GHOST_H in the script, so arrows meet the edges */
    width: 170px;
    height: 32px;
    box-sizing: border-box;
    border: 1.5px dashed var(--border-strong);
    border-radius: 7px;
    background: var(--card-bg);
    color: var(--muted);
    font-size: 12px;
    opacity: 0.6;
  }
  .suggestion:hover {
    opacity: 1;
  }
  .suggestion__label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-left: 6px;
    border-left: 3px solid #999;
  }
  .suggestion-wires {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: visible;
    pointer-events: none;
  }
  .suggestion-wires path {
    fill: none;
    stroke: var(--muted);
    stroke-width: 1.5;
    stroke-dasharray: 6 4;
    opacity: 0.65;
  }
  .suggestion-wires marker path {
    fill: var(--muted);
    stroke: none;
  }
  .suggestion__btn {
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    font-size: 10px;
    line-height: 1;
    padding: 3px 5px;
  }
  .suggestion__btn:hover {
    background: var(--hover-bg);
    color: var(--text);
  }
  .suggestion__btn--ok:hover {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .legend__toggle {
    cursor: pointer;
    margin-top: 4px;
  }
  .legend__toggle input {
    margin: 0;
  }
  .connect-menu__backdrop {
    position: fixed;
    inset: 0;
    z-index: 9;
  }
  .connect-menu {
    position: fixed;
    z-index: 10;
    width: 180px;
    max-height: 300px;
    overflow-y: auto;
    padding: 8px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--sidebar-bg);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  .connect-menu .palette__item:last-child {
    margin-bottom: 0;
  }
  .connect-menu__title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .connect-menu .hint {
    margin: 0;
  }
  .connect-menu__search {
    margin-bottom: 8px;
  }
  .deploy-dialog {
    width: min(560px, 90vw);
    padding: 16px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--sidebar-bg);
    color: var(--text);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  .deploy-dialog::backdrop {
    background: rgba(0, 0, 0, 0.4);
  }
  .deploy-dialog h3 {
    margin: 0 0 8px;
    font-size: 14px;
  }
  .deploy-dialog p {
    margin: 8px 0 4px;
    font-size: 12px;
    color: var(--muted-strong);
  }
  .deploy-dialog code {
    font-size: 11px;
    color: var(--text);
  }
  .deploy-dialog pre {
    margin: 0;
    padding: 8px 10px;
    font-size: 11px;
    overflow-x: auto;
    background: var(--input-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .deploy-dialog .action-btn {
    margin-top: 12px;
  }
  .dialog-hint {
    font-size: 11px;
    color: var(--muted);
  }
  /* The App node's page preview — inset so a strip of the editor stays
     visible behind it (the deployed page is the real fullscreen version).
     The 20px inset is mirrored in openApp's transform-origin math. */
  .app-overlay {
    position: fixed;
    inset: 20px;
    z-index: 20;
    overflow-y: auto;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
  }
  .app-overlay__close {
    position: fixed;
    top: 34px;
    right: 38px;
    z-index: 21;
    padding: 8px;
  }
  /* The recipe gallery: IS the empty state on a fresh canvas (transparent,
     canvas stays pannable around the cards), becomes a boxed overlay when
     reopened over existing work via the toolbar Tasks button. */
  .tasks {
    position: absolute;
    top: 44%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 6;
    width: min(640px, 80%);
    text-align: center;
    pointer-events: none;
  }
  .tasks--overlay {
    pointer-events: auto;
    background: var(--sidebar-bg);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    padding: 20px 24px 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
  }
  .tasks-backdrop {
    position: absolute;
    inset: 0;
    z-index: 5;
  }
  .tasks__title {
    font-size: 15px;
    font-weight: 600;
    color: var(--muted-strong);
    margin: 0 0 14px;
  }
  .tasks__grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .tasks__card {
    pointer-events: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    text-align: left;
    padding: 10px 12px;
    border: 1px solid var(--border-strong);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  }
  .tasks__card:hover {
    background: var(--hover-bg);
  }
  .tasks__card:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .tasks__name {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    font-weight: 600;
  }
  .tasks__name :global(svg) {
    flex: none;
    color: var(--accent);
  }
  .tasks__desc {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.35;
  }
  .tasks__alt {
    margin-top: 14px;
    font-size: 12px;
    color: var(--muted);
  }
  .tasks__chat {
    pointer-events: auto;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 6px;
    padding: 8px 12px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
  }
  .tasks__chat:hover {
    background: var(--hover-bg);
  }
  .tasks__chat :global(svg) {
    color: var(--accent);
  }
</style>
