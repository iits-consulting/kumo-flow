<script lang="ts">
  import Pager from "./Pager.svelte";
  import type { VolumeResult } from "./types";

  // Raw uint8 grayscale volumes, C-order (T,H,W), flat index t*H*W + y*W + x.
  // `volumes` are one per clip in the batch; urls already absolute (host prefixes API).
  // `seg` (same dims, same indexing) tints voxels with the instance's color.
  let { volumes }: { volumes: VolumeResult[] } = $props();
  let view = $state(0);
  let i = $derived(Math.min(view, volumes.length - 1));
  let vol = $derived(volumes[i]);

  // Fetched bytes for the selected volume. Guarded against a stale fetch
  // landing after the user already switched volumes.
  let data = $state<Uint8Array | null>(null);
  $effect(() => {
    const url = vol.url;
    data = null;
    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        if (vol.url === url) data = new Uint8Array(buf);
      });
  });

  let seg = $state<Uint8Array | null>(null);
  let overlay = $state(true);
  $effect(() => {
    const url = vol.seg;
    seg = null;
    if (!url) return;
    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        if (vol.seg === url) seg = new Uint8Array(buf);
      });
  });

  const AXES = ["axial", "coronal", "sagittal"] as const;
  type Axis = (typeof AXES)[number];
  let axis = $state<Axis>("axial");
  let len = $derived(axis === "axial" ? vol.dims[0] : axis === "coronal" ? vol.dims[1] : vol.dims[2]);
  let slice = $state(0);
  $effect(() => {
    slice = Math.floor(len / 2); // volume or axis changed — recenter
  });

  let canvas: HTMLCanvasElement | undefined = $state();
  $effect(() => {
    if (!canvas || !data) return;
    const [T, H, W] = vol.dims;
    const s = Math.min(slice, len - 1);
    const d = data;
    const sd = overlay ? seg : null;
    let w: number, h: number, idx: (x: number, y: number) => number;
    // ponytail: no voxel-spacing aspect correction — LoadVolume doesn't keep spacing; add when anisotropic volumes look too squashed
    if (axis === "axial") {
      [w, h] = [W, H];
      idx = (x, y) => s * H * W + y * W + x;
    } else if (axis === "coronal") {
      [w, h] = [W, T];
      idx = (x, row) => (T - 1 - row) * H * W + s * W + x;
    } else {
      [w, h] = [H, T];
      idx = (col, row) => (T - 1 - row) * H * W + col * W + s;
    }
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const k = idx(x, y);
        const o = (y * w + x) * 4;
        const g = d[k];
        const v = sd ? sd[k] : 0;
        const c = v ? (vol.colors?.[v - 1] ?? [255, 0, 0]) : null;
        img.data[o] = c ? (g + c[0]) >> 1 : g;
        img.data[o + 1] = c ? (g + c[1]) >> 1 : g;
        img.data[o + 2] = c ? (g + c[2]) >> 1 : g;
        img.data[o + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  });
</script>

{#if volumes.length > 1}
  <Pager count={volumes.length} bind:index={view} label="volume " />
{/if}
<div class="axes nodrag">
  {#each AXES as a}
    <button class:active={axis === a} onclick={() => (axis = a)}>{a}</button>
  {/each}
  {#if vol.seg}
    <button class:active={overlay} onclick={() => (overlay = !overlay)}>overlay</button>
  {/if}
</div>
<canvas class="preview nodrag" bind:this={canvas}></canvas>
{#if data}
  <input type="range" class="nodrag" min="0" max={len - 1} step="1" bind:value={slice} />
  <span class="muted">slice {slice}/{len}</span>
{/if}
{#if overlay && vol.names?.some(Boolean)}
  <span class="muted">
    {#each vol.names as n, j}
      {#if n}<span style="color: rgb({(vol.colors?.[j] ?? [255, 0, 0]).join(',')})">■</span> {n}{/if}
    {/each}
  </span>
{/if}

<style>
  .preview {
    max-width: var(--preview-max, 220px);
    max-height: var(--preview-max, 220px);
    border-radius: 4px;
    display: block;
  }
  .axes {
    display: flex;
    gap: 4px;
  }
  .axes button {
    padding: 3px 7px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 11px;
  }
  .axes button.active {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .muted {
    color: var(--muted);
    font-size: 12px;
  }
</style>
