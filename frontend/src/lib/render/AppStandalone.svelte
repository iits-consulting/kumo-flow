<script lang="ts">
  // Deployed host for AppView: the graph is a frozen snapshot behind deploy.py,
  // the spec comes from GET /info, and Run is one blocking POST /run (multipart,
  // field name = input node id) with a spinner — no per-node progress here.
  import AppView, { type AppPageSection } from "./AppView.svelte";
  import { toNodeResult, type NodeResult } from "./types";

  let title = $state("KumoFlow App");
  let sections = $state<AppPageSection[]>([]);
  let canRun = $state(false);
  let results = $state<Record<string, NodeResult>>({});
  let running = $state(false);
  let error = $state("");
  // Input node id -> chosen files. Not reactive: only read when Run posts.
  const files: Record<string, File[]> = {};

  async function load() {
    try {
      const info = await (await fetch("/info")).json();
      title = info.ui?.title || info.workflow || title;
      document.title = title;
      canRun = (info.targets ?? []).length > 0;
      // sections whose node vanished from the snapshot are skipped, never a crash
      sections = (info.ui?.sections ?? [])
        .filter((s: { node: string }) => info.kinds?.[s.node])
        .map((s: AppPageSection) => ({ ...s, kind: info.kinds[s.node] }));
    } catch (e) {
      error = `could not reach the pipeline API (${e})`;
    }
  }
  load();

  async function run() {
    running = true;
    error = "";
    try {
      const fd = new FormData();
      for (const [nid, fs] of Object.entries(files)) for (const f of fs) fd.append(nid, f);
      // no files picked = re-run on the deployed inputs, exactly like the API
      const res = await fetch("/run", { method: "POST", ...(Array.from(fd.keys()).length ? { body: fd } : {}) });
      const json = await res.json();
      if (!res.ok || !json.results) throw new Error(json.detail ?? json.error ?? `HTTP ${res.status}`);
      for (const [target, r] of Object.entries(json.results))
        results[target] = toNodeResult(r as Record<string, unknown>, "");
    } catch (e) {
      error = `run failed — ${e instanceof Error ? e.message : e}`;
    } finally {
      running = false;
    }
  }
</script>

{#if error}<p class="banner">{error}</p>{/if}
<AppView
  {title}
  {sections}
  {results}
  {running}
  {canRun}
  onRun={run}
  onFiles={(nid, _media, fs) => (files[nid] = fs)}
/>

<style>
  .banner {
    margin: 0;
    padding: 10px 24px;
    background: var(--danger);
    color: #fff;
    font-size: 13px;
  }
</style>
