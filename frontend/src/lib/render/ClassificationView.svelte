<script lang="ts">
  import Pager from "./Pager.svelte";
  import type { ClassResult } from "./types";

  // Bars for one image's classification, one ClassResult per batch item.
  let { items }: { items: ClassResult[] } = $props();
  let view = $state(0);
  let norm = $state(true); // normalized (sum-to-100%) vs raw sigmoid bars

  // Bars for one image's classification, sorted highest-first. `pct` is what the
  // number shows, `width` is the bar length (0..1).
  //  • normalized (≥2 labels) -> softmax over recovered logits (logit = sigmoid⁻¹); width = pct
  //  • raw / single label     -> pct is the true SigLIP sigmoid probability; SigLIP is
  //    calibrated so even strong matches sit at a few %, so the bar is log₁₀-scaled
  //    over [1e-6, 1] to make match vs no-match visible
  type ClassBar = { label: string; pct: number; width: number };
  const logWidth = (p: number) => Math.max(0, Math.min(1, 1 + Math.log10(Math.max(p, 1e-9)) / 6));
  function classBars(item: ClassResult, normalized: boolean): ClassBar[] {
    const { labels, scores } = item;
    let out: ClassBar[];
    if (!normalized) {
      out = labels.map((label, j) => ({ label, pct: scores[j], width: logWidth(scores[j]) }));
    } else {
      const clamp = (p: number) => Math.min(Math.max(p, 1e-6), 1 - 1e-6);
      const logits = scores.map((p) => Math.log(clamp(p) / (1 - clamp(p))));
      const m = Math.max(...logits);
      const ex = logits.map((l) => Math.exp(l - m));
      const s = ex.reduce((a, b) => a + b, 0);
      out = labels.map((label, j) => ({ label, pct: ex[j] / s, width: ex[j] / s }));
    }
    return out.sort((a, b) => b.pct - a.pct);
  }

  let i = $derived(Math.min(view, items.length - 1));
  let item = $derived(items[i]);
  let isNorm = $derived(norm && item.labels.length > 1);
  let rows = $derived(classBars(item, isNorm));
</script>

{#if item.labels.length > 1}
  <div class="cls-tools nodrag">
    <button class:active={norm} onclick={() => (norm = true)} title="sum-to-100% (softmax)">normalized</button>
    <button class:active={!norm} onclick={() => (norm = false)} title="true SigLIP sigmoid %, log-scaled bar">raw</button>
  </div>
{/if}
<div class="bars nodrag">
  {#each rows as row, j}
    <div class="bar-row" class:winner={j === 0}>
      <span class="bar-label" title={row.label}>{row.label}</span>
      <div class="bar-track" title={isNorm ? "" : "bar is log-scaled; the number is the true sigmoid %"}>
        <div class="bar-fill" style:width={`${row.width * 100}%`}></div>
      </div>
      <span class="bar-val">{(row.pct * 100).toFixed(isNorm ? 0 : 2)}%</span>
    </div>
  {/each}
</div>
{#if item.labels.length === 1}
  <p class="cls-hint">
    True SigLIP sigmoid % on a log-scaled bar — even strong matches score only a few %.
    Add comma-separated classes (or more Text Prompt nodes) to compare.
  </p>
{/if}
{#if items.length > 1}
  <Pager count={items.length} bind:index={view} />
{/if}

<style>
  .cls-tools {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  .cls-tools button {
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
  .cls-tools button.active {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 220px;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 72px 1fr 44px;
    align-items: center;
    gap: 6px;
  }
  .bar-label {
    font-size: 11px;
    text-align: right;
    color: var(--muted-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-track {
    background: var(--track-bg);
    border-radius: 3px;
    height: 14px;
    overflow: hidden;
  }
  .bar-fill {
    background: #fdba74; /* non-winner: muted orange */
    height: 100%;
    border-radius: 3px;
    min-width: 2px;
  }
  .bar-val {
    font-size: 10px;
    color: var(--muted);
  }
  .bar-row.winner .bar-fill {
    background: #ea580c; /* winner: full orange */
  }
  .bar-row.winner .bar-label {
    color: var(--text);
    font-weight: 600;
  }
  .cls-hint {
    margin: 0;
    font-size: 10px;
    line-height: 1.3;
    color: var(--muted);
  }
</style>
