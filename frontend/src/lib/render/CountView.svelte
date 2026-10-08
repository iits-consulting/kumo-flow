<script lang="ts">
  import type { CountResult } from "./types";

  let { counts }: { counts: CountResult } = $props();
</script>

{#if counts.per_image.length === 1 && counts.concepts.length === 1}
  <!-- one image, one concept: the answer is a number, not a table -->
  <div class="count-big">{counts.total[0]}<span class="count-unit">{counts.concepts[0]}</span></div>
{:else}
  <div class="count-wrap nodrag nowheel">
    <table class="count-table">
      <thead>
        <tr><th></th>{#each counts.concepts as name}<th title={name}>{name}</th>{/each}</tr>
      </thead>
      <tbody>
        {#each counts.per_image as row, i}
          <tr><td class="count-row-head">{i + 1}</td>{#each row as n}<td>{n}</td>{/each}</tr>
        {/each}
      </tbody>
      <tfoot>
        <tr><td class="count-row-head">total</td>{#each counts.total as n}<td>{n}</td>{/each}</tr>
      </tfoot>
    </table>
  </div>
{/if}

<style>
  .count-big {
    display: flex;
    align-items: baseline;
    justify-content: center;
    gap: 6px;
    font-size: 30px;
    font-weight: 600;
    padding: 4px 0;
  }
  .count-unit {
    font-size: 12px;
    font-weight: 400;
    color: var(--muted);
  }
  .count-wrap {
    max-height: 220px;
    overflow: auto;
  }
  .count-table {
    border-collapse: collapse;
    font-size: 11px;
    min-width: 160px;
  }
  .count-table th,
  .count-table td {
    padding: 2px 7px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .count-table thead th {
    position: sticky; /* a long batch scrolls under its column names */
    top: 0;
    background: var(--card-bg);
    color: var(--muted-strong);
    font-weight: 600;
    max-width: 70px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count-row-head {
    color: var(--muted);
    text-align: left !important;
  }
  .count-table tfoot td {
    border-top: 1px solid var(--border-strong);
    font-weight: 600;
  }
</style>
