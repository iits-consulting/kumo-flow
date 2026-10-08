<script lang="ts">
  // The app page: title, then sections top to bottom — upload widgets for
  // input sections, result panes for display sections — and one Run button.
  // One renderer, two hosts: the editor's fullscreen View (live graph,
  // FLOW_CTX.run) and the deployed bundle (frozen snapshot, POST /run). The
  // host resolves sections (kind, label fallback, missing nodes dropped) and
  // owns all fetching; this component only renders plain data.
  import ClassificationView from "./ClassificationView.svelte";
  import CountView from "./CountView.svelte";
  import ExportView from "./ExportView.svelte";
  import ImageView from "./ImageView.svelte";
  import ScatterView from "./ScatterView.svelte";
  import UploadWidget from "./UploadWidget.svelte";
  import VideoView from "./VideoView.svelte";
  import type { NodeResult } from "./types";

  export interface AppPageSection {
    node: string;
    kind: string;
    label: string;
    description?: string;
  }

  let {
    title,
    sections,
    results,
    running,
    canRun,
    progress = {},
    onRun,
    onFiles,
  }: {
    title: string;
    sections: AppPageSection[];
    results: Record<string, NodeResult>;
    running: boolean;
    canRun: boolean; // false = no display section to compute (Run disabled)
    progress?: Record<string, [number, number]>; // editor host only
    onRun: () => void;
    onFiles: (node: string, media: "image" | "video" | "volume", files: File[]) => void;
  } = $props();

  const INPUT_MEDIA: Record<string, "image" | "video" | "volume"> = { load: "image", load_video: "video", load_volume: "volume" };
</script>

<div class="app-page">
  <header>
    <h1>{title}</h1>
    <button class="run" onclick={onRun} disabled={running || !canRun}>
      {#if running}<span class="spinner"></span> Running…{:else}▶ Run{/if}
    </button>
  </header>
  {#each sections as s (s.node)}
    {@const r = results[s.node] ?? {}}
    {@const media = INPUT_MEDIA[s.kind]}
    {@const prog = progress[s.node]}
    <section>
      <h2>{s.label}</h2>
      {#if s.description}<p class="desc">{s.description}</p>{/if}
      {#if media}
        <UploadWidget {media} thumbs onFiles={(files) => onFiles(s.node, media, files)} />
      {:else if prog?.[1]}
        <div class="prog" title="{prog[0]} of {prog[1]} done">
          <div class="prog__fill" style:width={`${Math.min(100, (100 * prog[0]) / prog[1])}%`}></div>
          <span class="prog__num">{prog[0]}/{prog[1]}</span>
        </div>
      {:else if r.error}
        <p class="err">{r.error}</p>
      {:else if s.kind === "view_classification" && r.classification?.length}
        <ClassificationView items={r.classification} />
      {:else if s.kind === "count" && r.counts}
        <CountView counts={r.counts} />
      {:else if s.kind === "show_embeddings" && r.points?.length}
        <ScatterView points={r.points} thumbnails={r.images} />
      {:else if s.kind === "view_video" && r.videos?.length}
        <VideoView clips={r.videos} />
      {:else if s.kind === "export" && r.download}
        <ExportView download={r.download} />
      {:else if r.images?.length}
        <ImageView images={r.images} />
      {:else}
        <p class="empty">Press Run to see results.</p>
      {/if}
    </section>
  {/each}
  {#if !sections.length}
    <p class="empty">This app has no sections yet.</p>
  {/if}
</div>

<style>
  .app-page {
    --preview-max: min(100%, 640px);
    max-width: 720px;
    margin: 0 auto;
    padding: 32px 24px 64px;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
  }
  h1 {
    margin: 0;
    font-size: 22px;
  }
  section {
    margin-top: 28px;
  }
  h2 {
    margin: 0 0 4px;
    font-size: 15px;
  }
  .desc {
    margin: 0 0 10px;
    color: var(--muted);
    font-size: 12px;
  }
  .run {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 9px 22px;
    border: none;
    border-radius: 6px;
    background: var(--accent);
    color: var(--accent-text);
    cursor: pointer;
    font-size: 14px;
    white-space: nowrap;
  }
  .run:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid var(--accent-text);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .prog {
    position: relative;
    height: 14px;
    max-width: 320px;
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
    font-size: 13px;
  }
  .empty {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
  }
</style>
