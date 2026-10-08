<script lang="ts">
  import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/svelte";
  import { getContext } from "svelte";
  import { API, FLOW_CTX, filesToInput, handleStyle, PORT_COLORS, portColor, type FlowNodeData, type FlowContext } from "./flow";
  import Icon from "./Icon.svelte";
  import ClassificationView from "./render/ClassificationView.svelte";
  import CountView from "./render/CountView.svelte";
  import ExportView from "./render/ExportView.svelte";
  import ImageView from "./render/ImageView.svelte";
  import Pager from "./render/Pager.svelte";
  import ScatterView from "./render/ScatterView.svelte";
  import UploadWidget from "./render/UploadWidget.svelte";
  import VideoView from "./render/VideoView.svelte";
  import VolumeView from "./render/VolumeView.svelte";

  let { id, data, selected }: NodeProps = $props();
  let d = $derived(data as FlowNodeData);
  const flow = getContext<FlowContext>(FLOW_CTX);
  const updateNodeInternals = useUpdateNodeInternals();

  // Border communicates run state, overriding the node's category color:
  // pulsing (still its own color) while part of an in-flight run's subgraph,
  // then red/green once *this* node's own last run settles. Ancestor nodes
  // recomputed as a side effect of someone else's run only ever get the
  // in-progress pulse — the backend reports one pass/fail per request, not
  // per upstream node, so we can't attribute success/error to them.
  let isRunning = $derived(Boolean(flow.activeNodes[id]));
  let hasResult = $derived(
    Boolean(
      flow.results[id]?.length ||
        flow.classifications[id]?.length ||
        flow.counts[id] ||
        flow.points[id]?.length ||
        flow.videos[id]?.length ||
        flow.volumes[id]?.length,
    ),
  );
  let borderColor = $derived(
    isRunning ? d.color : flow.errors[id] ? "var(--danger)" : hasResult ? "var(--success)" : d.color,
  );

  // Picked files -> this node's config overlay (see filesToInput). Only the
  // video/volume path can fail (its multipart upload); images read locally.
  async function pick(files: File[], media: "image" | "video" | "volume") {
    try {
      flow.inputs[id] = await filesToInput(files, media);
    } catch (err) {
      flow.errors[id] = `upload failed — is the backend running on :8000? (${err})`;
    }
  }

  // "Cat×12, Dog×13" — what the folder pick labeled, shown on the Load node.
  function labelSummary(ls: string[]): string {
    const counts: Record<string, number> = {};
    for (const l of ls) if (l) counts[l] = (counts[l] ?? 0) + 1;
    return Object.entries(counts)
      .map(([k, v]) => `${k}×${v}`)
      .join(", ");
  }

  // LogReg: read an uploaded .txt/.csv into the labels config string
  // (one label per line or comma-separated; backend parses).
  async function pickLabels(e: Event) {
    const f = (e.currentTarget as HTMLInputElement).files?.[0];
    if (f) edit("labels", (await f.text()).trim());
  }

  function edit(key: string, value: unknown) {
    flow.inputs[id] = { ...flow.inputs[id], [key]: value };
  }

  // Live HF Hub search for Custom… model ids (fields with backend hf_search
  // tags): one query per pipeline tag, merged and sorted by downloads. The
  // Hub API is CORS-open, so the browser hits it directly — no proxy.
  let hfKey = $state(""); // which config field's results are showing
  let hfResults = $state<{ id: string; downloads: number }[]>([]);
  let hfTimer: ReturnType<typeof setTimeout>;
  function hfSearch(key: string, q: string) {
    hfKey = key;
    clearTimeout(hfTimer);
    hfTimer = setTimeout(async () => {
      const pages = await Promise.all(
        (d.config_hf?.[key] ?? []).map((tag) =>
          fetch(
            `https://huggingface.co/api/models?pipeline_tag=${tag}&library=transformers&sort=downloads&limit=8&search=${encodeURIComponent(q)}`,
          )
            .then((r) => (r.ok ? r.json() : []))
            .catch(() => []),
        ),
      );
      const seen = new Set<string>();
      hfResults = (pages.flat() as { id: string; downloads: number }[])
        .filter((m) => !seen.has(m.id) && Boolean(seen.add(m.id)))
        .sort((a, b) => b.downloads - a.downloads)
        .slice(0, 8);
    }, 300);
  }
  const hfCount = new Intl.NumberFormat("en", { notation: "compact" });

  // Fetch the image wired into this node's `image` port — or the `video` port
  // (the backend previews every frame of every clip, flattened clip-major, so
  // the pager steps through frames) — so the user can draw on it
  // (Visual Prompt points/boxes, Crop's and Filter by Region's box).
  function loadInput() {
    const src = flow.upstream(id, "image") ?? flow.upstream(id, "video");
    if (!src) {
      flow.errors[id] = "connect an image first";
      return;
    }
    flow.run(id, src);
  }

  const cfg = (key: string) => flow.inputs[id]?.[key] ?? d.config[key];

  // Custom Code nodes pick their ports (from the same fixed set every other
  // node uses — see PORT_COLORS) in the settings form below, so handles are
  // derived from that config instead of d.inputs/d.outputs.
  const asPortList = (v: unknown) => (Array.isArray(v) ? (v as string[]) : []);
  let inPorts = $derived(d.kind === "custom" ? asPortList(cfg("input_ports")) : d.inputs);
  let outPorts = $derived(d.kind === "custom" ? asPortList(cfg("output_ports")) : d.outputs);
  const isRequiredPort = (port: string) => (d.kind === "custom" ? true : d.required_inputs.includes(port));

  // Handles added/removed after a node's first render aren't picked up by
  // Svelte Flow's connection logic on their own — without this, dragging an
  // edge onto a newly-added Custom Code handle silently misses and falls
  // through to the "drop on empty canvas" behavior instead of connecting.
  $effect(() => {
    if (d.kind !== "custom") return;
    inPorts.length; outPorts.length; // reactive deps: rerun when ports change
    updateNodeInternals(id);
  });

  // Toggle one port name in/out of a Custom Code port-list field; the list
  // stays in canonical port order (PORT_COLORS key order) like every other node.
  function togglePort(key: string, port: string, on: boolean) {
    const cur = asPortList(cfg(key));
    const next = on ? [...cur, port] : cur.filter((p) => p !== port);
    edit(key, Object.keys(PORT_COLORS).filter((p) => next.includes(p)));
  }

  // Horizontal flow: handles sit on the left/right edges, so multiple ports
  // stack vertically along that edge instead of horizontally along the top/bottom.
  const stack = (i: number, n: number) => `top:${((i + 1) * 100) / (n + 1)}%`;

  // [work units done, total] while the backend is computing this node — only the
  // slow ones report (models, tracking, video decode), so absent = no bar.
  let prog = $derived(flow.progress[id]);

  let view = $state(0); // batch image the Visual Prompt surface currently shows/draws on

  // --- Visual Prompt drawing surface --------------------------------------
  // Positive/negative points and boxes drawn on the node's input image (fetched
  // via /run) in full-resolution pixel coords, stored back onto the node config.
  type Tool = "pt+" | "pt-" | "box+" | "box-";
  let tool = $state<Tool>("pt+");
  let img = $state<HTMLImageElement>();
  let drag = $state<{ x: number; y: number } | null>(null);
  let cur = $state<{ x: number; y: number } | null>(null);

  // Nodes whose drawing surface is a single rect stored in the "box" config
  // field (Crop's crop box, Filter by Region's region), drawn on batch item 0.
  let boxDraw = $derived(d.kind === "crop" || d.kind === "filter_region");

  // Per-surface prompt sets: frames[i] holds the points/boxes drawn on batch
  // image i, or (for video) the true flattened frame i across all clips —
  // NOT the preview page (see trueIdx below), which can be capped/subsampled.
  type VpFrame = { points?: number[][]; point_labels?: number[]; boxes?: number[][]; box_labels?: number[] };
  let vpImgs = $derived((flow.results[id] ?? []) as string[]); // the batch fetched via the load button
  let vpIdx = $derived(Math.min(view, Math.max(0, vpImgs.length - 1))); // preview page currently shown/drawn on
  // Preview page -> true flattened frame index: identity, unless the backend
  // capped a video-flattened preview (frame_indices), in which case page i
  // stands in for a frame further out — prompts must key off the true index,
  // the same one the backend regroups per_image by.
  let trueIdx = $derived(flow.frameIndices[id]?.[vpIdx] ?? vpIdx);
  let frames = $derived(((flow.inputs[id]?.frames ?? []) as VpFrame[]));
  let frame = $derived(frames[trueIdx] ?? {});
  let points = $derived((frame.points ?? []) as number[][]);
  let pointLabels = $derived((frame.point_labels ?? []) as number[]);
  let boxes = $derived((frame.boxes ?? []) as number[][]);
  let boxLabels = $derived((frame.box_labels ?? []) as number[]);
  // Size the drawing surface from the *source* image, not the <img>: previews
  // are downscaled to display size, so measuring the element would record every
  // box and point in preview pixels and the backend would read them as source
  // ones. Box-draw nodes always draw on the first batch item, the prompt
  // nodes on the paged one. Derived, so paging can't leave a stale scale behind.
  let natIdx = $derived(boxDraw ? 0 : vpIdx);
  let nat = $derived.by(() => {
    const s = flow.sizes[id]?.[natIdx];
    return { w: s?.[0] ?? 0, h: s?.[1] ?? 0 };
  });
  let r = $derived(Math.max(3, nat.w * 0.01)); // point radius in image px

  const POS = "#16a34a";
  const NEG = "#dc2626";

  // Merge a patch into the current image's frame (growing the array as needed).
  function setFrame(patch: VpFrame) {
    const fr = [...frames];
    while (fr.length <= trueIdx) fr.push({ points: [], point_labels: [], boxes: [], box_labels: [] });
    fr[trueIdx] = { ...fr[trueIdx], ...patch };
    flow.inputs[id] = { ...(flow.inputs[id] ?? {}), frames: fr };
  }
  function clampXY(e: PointerEvent) {
    const box = img!.getBoundingClientRect();
    const x = Math.round(((e.clientX - box.left) / box.width) * nat.w);
    const y = Math.round(((e.clientY - box.top) / box.height) * nat.h);
    return { x: Math.min(Math.max(x, 0), nat.w), y: Math.min(Math.max(y, 0), nat.h) };
  }
  function down(e: PointerEvent) {
    if (!nat.w) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    drag = cur = clampXY(e);
  }
  function move(e: PointerEvent) {
    if (drag) cur = clampXY(e);
  }
  function up(e: PointerEvent) {
    if (!drag) return;
    const p = clampXY(e);
    const x1 = Math.min(drag.x, p.x), y1 = Math.min(drag.y, p.y);
    const x2 = Math.max(drag.x, p.x), y2 = Math.max(drag.y, p.y);
    if (boxDraw) {
      if (x2 - x1 > 2 && y2 - y1 > 2) edit("box", `${x1},${y1},${x2},${y2}`);
    } else if (tool === "pt+" || tool === "pt-") {
      setFrame({ points: [...points, [p.x, p.y]], point_labels: [...pointLabels, tool === "pt+" ? 1 : 0] });
    } else {
      if (x2 - x1 > 2 && y2 - y1 > 2)
        setFrame({ boxes: [...boxes, [x1, y1, x2, y2]], box_labels: [...boxLabels, tool === "box+" ? 1 : 0] });
    }
    drag = cur = null;
  }
  const TOOLS: [Tool, string][] = [["pt+", "+pt"], ["pt-", "−pt"], ["box+", "+box"], ["box-", "−box"]];

  // Custom Code's "code" field: input/output shape cheat-sheet, tucked behind
  // a "?" instead of living as a wall of comments in the default code.
  let showCodeHelp = $state(false);
  const CODE_HELP = `inputs: one entry per input port picked below, e.g. inputs["image"].

Port shapes (batches are always a list, length B):
  image, overlay -> list of (3,H,W) uint8 tensors, RGB
  video          -> list of (T,3,H,W) uint8 tensors, one clip per item
  masks          -> list of (N,H,W) uint8 tensors, per-image instance masks
  boxes          -> list of (N,4) int32 tensors, x1,y1,x2,y2 inclusive
                    (from a video segmentation, masks/boxes items are instead
                    per-clip lists of T such tensors, one per frame)
  embedding      -> list of (D,) float32 tensors, one vector per image
  labels         -> list of str, one per image ("" = unlabeled); from SAM3
                    instead one list of concept names per image, aligned
                    1:1 with that image's boxes/masks
  ids            -> per-clip lists of T (N,) int32 tensors: the track id of
                    each box in that frame (-1 = flicker), from Track Objects
  prompts        -> dict, e.g. {"text": "cat"} or {"per_image": [...]}
  classification -> list of {"labels": [...], "scores": [...]}, one per image

return a dict with one entry per output port picked below, same shapes.

optional: define a top-level load() — it runs once per code version (cached;
editing the code reloads) and its return is passed to run as the first
argument, run(loaded, **inputs). Load your own HF model there.`;

  // Crop's drawn box, parsed from the "x1,y1,x2,y2" config string.
  let cropBox = $derived.by(() => {
    const c = String(cfg("box") ?? "").split(",").map(Number);
    return c.length === 4 && c.every((n) => !isNaN(n)) ? c : null;
  });

  // Progressive reveal: generic nodes (no dedicated body) hide their config
  // behind a click-to-open settings popover and show only a value summary.
  const SPECIAL = ["load", "load_video", "crop", "filter_region", "visual_prompt", "text_prompt", "view_classification", "note"];

  // View nodes' legend: one chip per detected object (label / track id /
  // instance index), showing the color it's drawn in. Picking a color stores
  // it in the node's `colors` config and re-runs the node to re-bake overlays.
  let pickedColors = $derived((cfg("colors") as Record<string, string> | undefined) ?? {});
  function setColor(key: string, value: string) {
    edit("colors", { ...pickedColors, [key]: value });
  }
  function clearColor(key: string) {
    const { [key]: _, ...rest } = pickedColors;
    edit("colors", rest);
    flow.run(id);
  }
  // Custom Code has no fixed outputs (d.outputs is always [] — ports are
  // typed in, not declared on the class), so it needs an explicit opt-in.
  let generic = $derived(!SPECIAL.includes(d.kind) && (d.outputs.length > 0 || d.kind === "custom"));
  let hasCfg = $derived(Object.keys(d.config).length > 0);
  // Config opens in the settings popover for generic nodes, plus upload nodes
  // whose remaining knobs (fps, …) shouldn't crowd the card next to the file
  // button; the popover skips the upload payload key itself ("data").
  let openable = $derived((generic || d.kind === "load_video") && hasCfg);
  let showSettings = $state(false);
  // Custom Code's config is source code, not a glanceable value — never preview it.
  let summary = $derived(
    d.kind === "custom"
      ? ""
      : Object.keys(d.config)
          .map((k) => String(cfg(k) ?? ""))
          .filter(Boolean)
          .join(", ")
          .slice(0, 120), // an uploaded labels file can be huge; the summary is a glance, not the data
  );
  // Clicking elsewhere deselects the node — close the popover with it.
  $effect(() => {
    if (!selected) showSettings = false;
  });

  // A run failed because this prompt node is empty (flow.attention bumped):
  // open for input — Text Prompt focuses its always-visible text field,
  // Visual Prompt fetches its upstream image as if its load button was pressed.
  let promptInput = $state<HTMLInputElement>();
  let seenAttention = 0;
  $effect(() => {
    const n = flow.attention[id] ?? 0;
    if (n <= seenAttention) return;
    seenAttention = n;
    if (d.kind === "visual_prompt") loadInput();
    else promptInput?.focus();
  });
</script>

{#each inPorts as port, i}
  <Handle
    type="target"
    position={Position.Left}
    id={port}
    title={`${port}${isRequiredPort(port) ? "" : " (optional)"}`}
    style={`${stack(i, inPorts.length)};${handleStyle(port, isRequiredPort(port))}`}
  />
{/each}

<!-- Every output node's ▶ Run, which becomes "Running…" next to a Stop button
     while the run is in flight. Stop cancels only this node's target — upstream
     work shared with other running targets keeps going (canvas-wide stop lives
     in the bottom panel). -->
{#snippet runRow()}
  <div class="run-row">
    <button class="run" onclick={() => flow.run(id)} disabled={flow.running[id]}>
      {#if flow.running[id]}Running…{:else}<Icon name="play" size={10} /> Run{/if}
    </button>
    {#if flow.running[id]}
      <button class="stop" onclick={() => flow.stop(id)}>
        {#if flow.stopping}Stopping…{:else}<Icon name="stop" size={10} /> Stop{/if}
      </button>
    {/if}
  </div>
{/snippet}

<!-- The View nodes' object legend: one chip per detected object, thumbnail +
     key + the color it's drawn in. Picking a color (or × back to auto) re-runs
     the node — overlays are baked server-side, so a change needs a re-render. -->
{#snippet legendRow()}
  {#if flow.legends[id]?.length}
    <div class="legend nodrag">
      {#each flow.legends[id] as e (e.key)}
        <label class="legend-chip" title={`color of ${e.key}`}>
          {#if e.thumb}<img src={e.thumb} alt={e.key} />{/if}
          <span class="legend-key">{e.key}</span>
          <!-- a fresh pick wins over the last run's drawn color, so re-renders
               mid-pick can't snap the input back -->
          <input
            type="color"
            value={pickedColors[e.key] ?? e.color}
            oninput={(ev) => setColor(e.key, (ev.currentTarget as HTMLInputElement).value)}
            onchange={() => flow.run(id)}
          />
          {#if pickedColors[e.key]}
            <button title="back to auto color" onclick={(ev) => { ev.preventDefault(); clearColor(e.key); }}>×</button>
          {/if}
        </label>
      {/each}
      {#if flow.legendMore[id]}
        <span class="legend-more">+{flow.legendMore[id]} more</span>
      {/if}
    </div>
  {/if}
{/snippet}

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div
  class="flow-node"
  class:is-running={isRunning}
  style:border-color={borderColor}
  onclick={openable ? () => (showSettings = !showSettings) : undefined}
>
  <div class="flow-node__title" style:background={d.color}>
    {d.label}
    {#if openable}<span class="gear"><Icon name="sliders" size={11} /></span>{/if}
  </div>
  {#if !generic || summary || flow.errors[id] || prog?.[1]}
  <div class="flow-node__body">
    {#if prog?.[1]}
      <!-- clamped, not trusted: a node that ever ticks past its declared total
           should overflow the label, not the bar -->
      <div class="prog" title="{prog[0]} of {prog[1]} done">
        <div class="prog__fill" style:width={`${Math.min(100, (100 * prog[0]) / prog[1])}%`}></div>
        <span class="prog__num">{prog[0]}/{prog[1]}</span>
      </div>
    {/if}
    {#if d.kind === "load" || d.kind === "load_video" || d.kind === "load_volume"}
      {@const media = d.kind === "load" ? "image" : d.kind === "load_video" ? "video" : "volume"}
      <UploadWidget {media} onFiles={(files) => pick(files, media)} />
      {#if flow.inputs[id]?.name}<span class="muted">{String(flow.inputs[id].name)}</span>{/if}
      {#if (flow.inputs[id]?.labels as string[] | undefined)?.some(Boolean)}
        <span class="muted">labels: {labelSummary(flow.inputs[id].labels as string[])}</span>
      {/if}
    {:else if boxDraw}
      {@const boxName = d.kind === "crop" ? "crop box" : "region"}
      <div class="vp-tools nodrag">
        <button onclick={loadInput} disabled={flow.running[id]} title="load input image"><Icon name="refresh" size={10} /></button>
        <button onclick={() => edit("box", "")}>clear</button>
      </div>
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.results[id]?.length}
        <div
          class="vp-wrap nodrag"
          role="application"
          onpointerdown={down}
          onpointermove={move}
          onpointerup={up}
        >
          <img
            bind:this={img}
            src={flow.results[id][0]}
            alt="draw {boxName}"
            draggable="false"
          />
          {#if nat.w}
            <svg viewBox={`0 0 ${nat.w} ${nat.h}`} preserveAspectRatio="none">
              {#if drag && cur}
                <rect
                  x={Math.min(drag.x, cur.x)} y={Math.min(drag.y, cur.y)}
                  width={Math.abs(cur.x - drag.x)} height={Math.abs(cur.y - drag.y)}
                  fill="none" stroke={POS} stroke-width={r / 2} stroke-dasharray={r}
                />
              {:else if cropBox}
                <rect
                  x={cropBox[0]} y={cropBox[1]} width={cropBox[2] - cropBox[0]} height={cropBox[3] - cropBox[1]}
                  fill="none" stroke={POS} stroke-width={r / 2}
                />
              {/if}
            </svg>
          {/if}
        </div>
        {#if cropBox}<span class="muted">{boxName}: {cropBox.join(", ")}</span>{/if}
      {:else}
        <span class="muted">connect an image or video, then <Icon name="refresh" size={9} /> to draw the {boxName}</span>
      {/if}
    {:else if d.kind === "visual_prompt"}
      <div class="vp-tools nodrag">
        {#each TOOLS as [t, lbl]}
          <button class:active={tool === t} onclick={() => (tool = t)}>{lbl}</button>
        {/each}
        <button onclick={loadInput} disabled={flow.running[id]} title="load image"><Icon name="refresh" size={10} /></button>
        <button onclick={() => setFrame({ points: [], point_labels: [], boxes: [], box_labels: [] })}
          title="clear this image's prompts">clear</button>
      </div>
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if vpImgs.length}
        <div
          class="vp-wrap nodrag"
          role="application"
          onpointerdown={down}
          onpointermove={move}
          onpointerup={up}
        >
          <img
            bind:this={img}
            src={vpImgs[vpIdx]}
            alt="draw prompts"
            draggable="false"
          />
          {#if nat.w}
            <svg viewBox={`0 0 ${nat.w} ${nat.h}`} preserveAspectRatio="none">
              {#each boxes as b, i}
                <rect
                  x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]}
                  fill="none" stroke={boxLabels[i] ? POS : NEG} stroke-width={r / 2}
                />
              {/each}
              {#if drag && cur && (tool === "box+" || tool === "box-")}
                <rect
                  x={Math.min(drag.x, cur.x)} y={Math.min(drag.y, cur.y)}
                  width={Math.abs(cur.x - drag.x)} height={Math.abs(cur.y - drag.y)}
                  fill="none" stroke={tool === "box+" ? POS : NEG} stroke-width={r / 2} stroke-dasharray={r}
                />
              {/if}
              {#each points as p, i}
                <circle cx={p[0]} cy={p[1]} r={r} fill={pointLabels[i] ? POS : NEG} stroke="#fff" stroke-width={r / 3} />
              {/each}
            </svg>
          {/if}
        </div>
        {#if vpImgs.length > 1}
          <Pager count={vpImgs.length} bind:index={view} />
        {/if}
      {:else}
        <span class="muted">connect an image or video, then <Icon name="refresh" size={9} /> to load it</span>
      {/if}
    {:else if d.kind === "view_classification"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.classifications[id]?.length}
        <ClassificationView items={flow.classifications[id]} />
      {/if}
    {:else if d.kind === "count"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.counts[id]}
        <CountView counts={flow.counts[id]} />
      {/if}
    {:else if d.kind === "show_embeddings"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.points[id]?.length}
        <ScatterView points={flow.points[id]} thumbnails={flow.results[id]} />
      {/if}
    {:else if d.kind === "view_video"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.videos[id]?.length}
        <VideoView clips={flow.videos[id].map((c) => API + c)} />
        {@render legendRow()}
      {/if}
    {:else if d.kind === "view_volume"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.volumes[id]?.length}
        <VolumeView volumes={flow.volumes[id].map((v) => ({ ...v, url: API + v.url, seg: v.seg && API + v.seg }))} />
      {/if}
    {:else if d.kind === "text_prompt"}
      <!-- the text IS the node — edited in place, never tucked into settings -->
      <input
        class="path nodrag"
        bind:this={promptInput}
        placeholder={'e.g. "person, truck"'}
        value={String(cfg("text") ?? "")}
        oninput={(e) => edit("text", (e.currentTarget as HTMLInputElement).value)}
      />
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
    {:else if d.kind === "note"}
      <!-- editable in place: a note is all body, no run button and no ports -->
      <textarea
        class="note nodrag nowheel"
        placeholder="Notes for this workflow…"
        value={String(cfg("text") ?? "")}
        oninput={(e) => edit("text", (e.currentTarget as HTMLTextAreaElement).value)}
      ></textarea>
    {:else if d.kind === "export"}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.downloads[id]}
        <ExportView download={flow.downloads[id]} />
      {:else}
        <span class="muted">wire any outputs in, then Run — downloads everything as one zip</span>
      {/if}
    {:else if d.kind !== "custom" && d.outputs.length === 0}
      {@render runRow()}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if flow.results[id]?.length}
        <ImageView images={flow.results[id]} />
        {@render legendRow()}
      {/if}
    {:else}
      {#if flow.errors[id]}<p class="err">{flow.errors[id]}</p>{/if}
      {#if summary}<div class="summary muted" title={summary}>{summary}</div>{/if}
    {/if}
  </div>
  {/if}
</div>

{#if showSettings}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div
    class="settings nodrag nowheel"
    style:width={d.kind === "custom" ? "340px" : "220px"}
    onclick={(e) => e.stopPropagation()}
  >
    <div class="settings__head">
      <span>{d.label} settings</span>
      <button class="settings__close" onclick={() => (showSettings = false)} aria-label="Close settings">
        <Icon name="x" size={11} />
      </button>
    </div>
    <!-- one control per config field; pydantic validates backend-side. Upload
         payloads are never typed settings: "data" (data URLs) and list-valued
         "labels" (per-item folder labels — LogReg's is a typed string, kept) -->
    {#snippet info(key: string)}
      <!-- hyperparameter help from the backend Field description; native title tooltip -->
      {#if d.config_info?.[key]}<span class="info" title={d.config_info[key]}><Icon name="info" size={11} /></span>{/if}
    {/snippet}
    {#each Object.keys(d.config).filter((k) => k !== "data" && !(k === "labels" && Array.isArray(d.config[k]))) as key}
      {#if d.config_options?.[key]}
        <!-- dropdown presets from the backend Field options; dict options get a
             Custom… entry that reveals a free-text input (arbitrary HF model ids) -->
        {@const opts = d.config_options[key].map((o) => (typeof o === "string" ? { value: o, label: o } : o))}
        {@const custom = d.config_options[key].some((o) => typeof o !== "string")}
        {@const val = String(cfg(key) ?? "")}
        {@const preset = opts.some((o) => o.value === val)}
        <label class="muted">
          {key}{@render info(key)}
          <select
            class="path"
            value={preset ? val : "__custom__"}
            onchange={(e) => {
              const v = (e.currentTarget as HTMLSelectElement).value;
              edit(key, v === "__custom__" ? "" : v);
            }}
          >
            {#each opts as o}<option value={o.value}>{o.label}</option>{/each}
            {#if custom}<option value="__custom__">Custom…</option>{/if}
          </select>
          {#if custom && !preset}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div onfocusout={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) hfKey = ""; }}>
              <input
                class="path"
                placeholder={d.config_hf?.[key] ? "search the HF Hub…" : "model id, e.g. org/name"}
                value={val}
                oninput={(e) => {
                  const v = (e.currentTarget as HTMLInputElement).value;
                  edit(key, v);
                  if (d.config_hf?.[key]) hfSearch(key, v);
                }}
                onfocus={() => d.config_hf?.[key] && hfSearch(key, val)}
              />
              {#if hfKey === key && hfResults.length}
                <div class="hf-results nowheel">
                  {#each hfResults as m (m.id)}
                    <button type="button" class="hf-result" onclick={() => { edit(key, m.id); hfKey = ""; }}>
                      <span class="hf-id" title={m.id}>{m.id}</span>
                      <span class="muted">{hfCount.format(m.downloads)}↓</span>
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </label>
      {:else if typeof d.config[key] === "boolean"}
        <label class="muted toggle">
          <input
            type="checkbox"
            checked={Boolean(cfg(key))}
            onchange={(e) => edit(key, (e.currentTarget as HTMLInputElement).checked)}
          />
          {key}{@render info(key)}
        </label>
      {:else if key === "threshold" || key.endsWith("_threshold")}
        <label class="muted">
          {key}: {Number(cfg(key)).toFixed(2)}{@render info(key)}
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={Number(cfg(key))}
            oninput={(e) => edit(key, Number((e.currentTarget as HTMLInputElement).value))}
          />
        </label>
      {:else if key === "input_ports" || key === "output_ports"}
        {@const chosen = asPortList(cfg(key))}
        <div class="muted">
          {key}
          <div class="port-picker">
            {#each Object.keys(PORT_COLORS) as p}
              {@const on = chosen.includes(p)}
              <label class="port-chip" class:selected={on} style:border-color={portColor(p)} style:background={on ? portColor(p) : "transparent"}>
                <input type="checkbox" checked={on} onchange={(e) => togglePort(key, p, (e.currentTarget as HTMLInputElement).checked)} />
                {p}
              </label>
            {/each}
          </div>
        </div>
      {:else if key === "code"}
        <label class="muted">
          <span class="code-label">
            {key}
            <button
              type="button"
              class="help-btn"
              title="input/output shapes"
              onclick={(e) => { e.preventDefault(); showCodeHelp = !showCodeHelp; }}
            >?</button>
          </span>
          {#if showCodeHelp}<pre class="code-help">{CODE_HELP}</pre>{/if}
          <textarea
            class="code"
            rows="10"
            spellcheck="false"
            value={String(cfg(key) ?? "")}
            oninput={(e) => edit(key, (e.currentTarget as HTMLTextAreaElement).value)}
          ></textarea>
        </label>
      {:else}
        <label class="muted">
          {key}{@render info(key)}
          <input
            class="path"
            value={String(cfg(key) ?? "")}
            oninput={(e) => edit(key, (e.currentTarget as HTMLInputElement).value)}
          />
        </label>
      {/if}
    {/each}
    {#if d.kind === "logreg"}
      <label class="file-btn nodrag">Load labels file (.txt / .csv)
        <input type="file" accept=".txt,.csv,text/plain,text/csv" onchange={pickLabels} hidden />
      </label>
      <span class="muted">one label per line (or "file,label" rows), batch order; used when no labels port is wired</span>
    {/if}
  </div>
{/if}

{#each outPorts as port, i}
  <Handle
    type="source"
    position={Position.Right}
    id={port}
    title={port}
    style={`${stack(i, outPorts.length)};${handleStyle(port, true)}`}
  />
{/each}

<style>
  .flow-node {
    border: 2px solid #999;
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    min-width: 160px;
    font-size: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .flow-node.is-running {
    animation: node-pulse 1.1s ease-in-out infinite;
  }
  @keyframes node-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.5); }
    50% { box-shadow: 0 0 0 6px rgba(99, 102, 241, 0); }
  }
  .flow-node__title {
    color: #fff;
    font-weight: 600;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .gear {
    display: flex;
    align-items: center;
    opacity: 0.8;
  }
  /* icon-in-text and icon-in-button alignment: svgs are inline-block by
     default and ride the text baseline, which looks sunken at these sizes */
  .muted :global(svg) {
    vertical-align: -1px;
  }
  .flow-node__body {
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .summary {
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .settings {
    position: absolute;
    left: calc(100% + 14px);
    top: 0;
    width: 220px;
    padding: 10px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    cursor: default;
  }
  .settings__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
  }
  .settings__close {
    border: none;
    background: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 12px;
    padding: 0;
  }
  .path {
    width: 100%;
    box-sizing: border-box;
    padding: 5px 7px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    color: var(--text);
    font-size: 12px;
  }
  .muted {
    color: var(--muted);
  }
  .hf-results {
    margin-top: 4px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    max-height: 180px;
    overflow-y: auto;
  }
  .hf-result {
    display: flex;
    width: 100%;
    gap: 6px;
    align-items: center;
    justify-content: space-between;
    padding: 4px 7px;
    border: none;
    background: none;
    color: var(--text);
    font-size: 11px;
    cursor: pointer;
    text-align: left;
  }
  .hf-result:hover {
    background: var(--input-bg);
  }
  .hf-id {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .code-label {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .info {
    margin-left: 4px;
    cursor: help;
  }
  .help-btn {
    width: 15px;
    height: 15px;
    line-height: 13px;
    padding: 0;
    border: 1px solid var(--border-strong);
    border-radius: 50%;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    font-size: 10px;
  }
  .help-btn:hover {
    background: var(--hover-bg);
  }
  .code-help {
    margin: 4px 0;
    padding: 8px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    color: var(--text);
    font-family: ui-monospace, monospace;
    font-size: 10px;
    line-height: 1.4;
    white-space: pre-wrap;
  }
  .code {
    width: 100%;
    box-sizing: border-box;
    padding: 5px 7px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    color: var(--text);
    font-family: ui-monospace, monospace;
    font-size: 11px;
    resize: vertical;
  }
  /* resize: both — the note grows with the node, that's the whole sizing story */
  .note {
    width: 220px;
    /* grows to its text so a recipe's guide note is readable without dragging;
       min/max are the floor for an empty note and the ceiling before it scrolls */
    field-sizing: content;
    min-height: 90px;
    max-height: 420px;
    box-sizing: border-box;
    padding: 5px 7px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    color: var(--text);
    font: inherit;
    resize: both;
  }
  .file-btn {
    display: block;
    padding: 5px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    text-align: center;
    cursor: pointer;
  }
  .file-btn:hover {
    background: var(--hover-bg);
  }
  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
  }
  input[type="range"] {
    width: 100%;
    display: block;
  }
  .port-picker {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }
  .port-chip {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 7px;
    border: 1.5px solid;
    border-radius: 10px;
    cursor: pointer;
    font-size: 11px;
    color: var(--text);
  }
  .port-chip input {
    margin: 0;
  }
  .port-chip.selected {
    color: #fff;
  }
  .run {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 5px 10px;
    border: none;
    border-radius: 4px;
    background: var(--accent);
    color: var(--accent-text);
    cursor: pointer;
    font-size: 12px;
  }
  .run:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .run-row {
    display: flex;
    gap: 4px;
  }
  .run-row .run {
    flex: 1;
  }
  .stop {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 5px 10px;
    border: 1px solid var(--danger);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--danger);
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
  }
  /* Work units the node reports as it goes (frames, images, clips) — the
     pulsing border says "busy", this says how far along. */
  .prog {
    position: relative;
    height: 14px;
    border-radius: 3px;
    background: var(--track-bg);
    overflow: hidden;
  }
  .prog__fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s linear;
  }
  .prog__num {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    color: var(--muted-strong);
  }
  .err {
    margin: 0;
    color: var(--danger);
    font-size: 11px;
  }
  .vp-tools {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    max-width: 280px;
  }
  .legend-chip {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 4px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text);
    cursor: pointer;
  }
  .legend-chip img {
    width: 22px;
    height: 22px;
    object-fit: cover;
    border-radius: 3px;
  }
  .legend-key {
    max-width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .legend-chip input[type="color"] {
    width: 26px;
    height: 20px;
    padding: 1px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    cursor: pointer;
  }
  .legend-more {
    font-size: 10px;
    color: var(--muted);
    padding: 2px 4px;
  }
  .legend-chip button {
    width: 16px;
    height: 16px;
    padding: 0;
    border: none;
    background: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 12px;
    line-height: 1;
  }
  .vp-tools button {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 3px 6px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 11px;
  }
  .vp-tools button.active {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .vp-wrap {
    position: relative;
    display: inline-block;
    line-height: 0;
    cursor: crosshair;
    touch-action: none;
  }
  .vp-wrap img {
    max-width: 280px;
    display: block;
    border-radius: 4px;
    user-select: none;
  }
  .vp-wrap svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
</style>
