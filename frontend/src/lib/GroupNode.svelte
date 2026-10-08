<script lang="ts">
  import type { NodeProps } from "@xyflow/svelte";

  let { data }: NodeProps = $props();
</script>

<div class="group-node">
  {#if data?.label}<div class="group-node__label">{data.label}</div>{/if}
</div>

<style>
  .group-node {
    position: relative;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    border: 2px dashed var(--border-strong);
    border-radius: 10px;
    /* translucent: a selected group gets elevated above other nodes by xyflow,
       and an opaque box would hide whatever its bounding box happens to span */
    background: var(--group-bg);
  }
  .group-node__label {
    position: absolute;
    top: 8px;
    left: 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted-strong);
  }
</style>
