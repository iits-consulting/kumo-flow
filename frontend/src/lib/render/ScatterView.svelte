<script lang="ts">
  // View Embeddings: scale raw [x, y] points into a fixed square viewBox (y up,
  // plot convention). Degenerate spans (all points equal) collapse to the center.
  // `thumbnails` (optional, one per point) show on hover.
  let { points, thumbnails }: { points: number[][]; thumbnails?: string[] } = $props();

  const SCATTER = 220; // ponytail: fixed 220px everywhere — hover-thumb offsets are in viewBox px, scaling needs a coord rework
  let hover = $state<number | null>(null); // scatter point under the pointer
  function scatter(pts: number[][], pad = 12) {
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
    const sx = (SCATTER - 2 * pad) / (Math.max(...xs) - x0 || 1);
    const sy = (SCATTER - 2 * pad) / (Math.max(...ys) - y0 || 1);
    return pts.map(([x, y]) => ({ x: pad + (x - x0) * sx, y: SCATTER - pad - (y - y0) * sy }));
  }
  let pts = $derived(scatter(points));
</script>

<div class="scatter-wrap nodrag">
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <svg class="scatter" viewBox={`0 0 ${SCATTER} ${SCATTER}`} onpointerleave={() => (hover = null)}>
    {#each pts as p, i}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <circle cx={p.x} cy={p.y} r="4" onpointerenter={() => (hover = i)}>
        {#if !thumbnails}<title>image {i + 1}</title>{/if}
      </circle>
    {/each}
  </svg>
  <!-- the svg is rendered at exactly SCATTER px, so viewBox coords are pixel
       coords; flip the preview to the other side of the point near an edge -->
  {#if hover !== null && thumbnails?.[hover]}
    <img
      class="thumb"
      src={thumbnails[hover]}
      alt={`image ${hover + 1}`}
      style:left={`${pts[hover].x + 10}px`}
      style:top={`${pts[hover].y + 10}px`}
      style:transform={`${pts[hover].x > SCATTER / 2 ? "translateX(calc(-100% - 20px))" : ""} ${pts[hover].y > SCATTER / 2 ? "translateY(calc(-100% - 20px))" : ""}`}
    />
  {/if}
</div>

<style>
  .scatter-wrap {
    position: relative;
    line-height: 0;
  }
  .scatter {
    width: 220px;
    height: 220px;
    background: var(--input-bg);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
  }
  .scatter circle {
    fill: #0d9488; /* = the embedding port color */
    stroke: #fff;
    stroke-width: 1;
  }
  .thumb {
    position: absolute;
    max-width: 96px;
    max-height: 96px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    pointer-events: none; /* never steal the hover from the point under it */
  }
</style>
