<script lang="ts">
  import Icon from "../Icon.svelte";

  // ‹ 2 / 5 › — pages a batch result. `index` is clamped for display so a
  // shrinking batch can't leave the label past the end.
  let { count, index = $bindable(0), label = "" }: { count: number; index?: number; label?: string } = $props();
  let i = $derived(Math.min(index, Math.max(0, count - 1)));
</script>

<div class="pager nodrag">
  <button onclick={() => (index = Math.max(0, i - 1))} disabled={i === 0} aria-label="previous">
    <Icon name="chevron-left" size={10} />
  </button>
  <span class="muted">{label}{i + 1} / {count}</span>
  <button onclick={() => (index = Math.min(count - 1, i + 1))} disabled={i === count - 1} aria-label="next">
    <Icon name="chevron-right" size={10} />
  </button>
</div>

<style>
  .pager {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .pager button {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 13px;
    line-height: 1;
  }
  .pager button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .muted {
    color: var(--muted);
    font-size: 12px;
  }
</style>
