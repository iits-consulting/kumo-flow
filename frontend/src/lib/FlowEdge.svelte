<script lang="ts">
  import { BaseEdge, EdgeLabel, getBezierPath, type EdgeProps } from "@xyflow/svelte";
  import { getContext } from "svelte";
  import { FLOW_CTX, type FlowContext } from "./flow";
  import Icon from "./Icon.svelte";

  let {
    id,
    target,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    sourceHandleId,
    targetHandleId,
    style,
    markerEnd,
  }: EdgeProps = $props();
  const flow = getContext<FlowContext>(FLOW_CTX);
  // [path, labelX, labelY] — the label point is where the + button sits
  let p = $derived(getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition }));
</script>

<BaseEdge {id} path={p[0]} {style} {markerEnd} />
{#if flow.canInsert(sourceHandleId ?? "image", target, targetHandleId ?? "image")}
  <EdgeLabel x={p[1]} y={p[2]} transparent>
    <button
      class="edge-insert"
      title="Insert a node here"
      aria-label="Insert a node into this edge"
      onclick={(e) => flow.insertOnEdge(id, { x: p[1], y: p[2] }, e)}
    >
      <Icon name="plus" size={10} />
    </button>
  </EdgeLabel>
{/if}

<style>
  .edge-insert {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    padding: 0;
    border: 1px solid var(--border-strong);
    border-radius: 50%;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    opacity: 0.4;
  }
  .edge-insert:hover {
    opacity: 1;
    color: var(--text);
  }
</style>
