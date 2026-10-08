<script lang="ts">
  // Rendered inside <SvelteFlow>, which is the only place useSvelteFlow()'s
  // context is reachable — App.svelte (SvelteFlow's parent) can't call it.
  import { useSvelteFlow } from "@xyflow/svelte";
  import { tick } from "svelte";
  import type { Deployment } from "./flow";
  import Icon from "./Icon.svelte";

  let {
    onSave,
    onLoad,
    onTasks,
    deployments,
    onStopDeployment,
    fitView = $bindable(),
  }: {
    onSave: () => void;
    onLoad: (file: File) => Promise<void>;
    onTasks: () => void;
    deployments: Deployment[];
    onStopDeployment: (name: string) => void;
    // handed up to App, whose bottom action bar can't reach the context itself
    fitView?: (opts?: { duration?: number }) => void;
  } = $props();
  fitView = useSvelteFlow().fitView;
  let fileInput: HTMLInputElement;
  let showLive = $state(false);
  let liveCount = $derived(deployments.filter((d) => d.alive).length);

  async function loadAndFit(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    await onLoad(file); // file read + state swap must land before fitting the view
    input.value = ""; // allow picking the same file again
    await tick();
    fitView?.({ duration: 200 });
  }
</script>

{#if deployments.length}
  <span class="live-anchor">
    <button class="panel-btn" onclick={() => (showLive = !showLive)} title="Running deployments — each is its own process with its own model copies">
      <Icon name="deploy" /> Live ({liveCount})
    </button>
    {#if showLive}
      <div class="live-pop">
        {#each deployments as d (d.name)}
          <div class="live-row">
            <a href={`http://localhost:${d.port}`} target="_blank" title={d.has_ui ? "app page + API" : "API"}>{d.name}</a>
            <span class="live-port" class:dead={!d.alive}>:{d.port}{d.alive ? "" : " (dead)"}</span>
            <button class="live-stop" onclick={() => onStopDeployment(d.name)}>Stop</button>
          </div>
        {/each}
      </div>
    {/if}
  </span>
{/if}
<button class="panel-btn" onclick={onTasks} title="Common tasks — pre-wired workflows for typical goals">
  <Icon name="tasks" /> Tasks
</button>
<button class="panel-btn" onclick={onSave} title="Download the workflow as a JSON file">
  <Icon name="save" /> Save
</button>
<button class="panel-btn" onclick={() => fileInput.click()} title="Load a workflow from a JSON file">
  <Icon name="load" /> Load
</button>
<input type="file" accept=".json,application/json" hidden bind:this={fileInput} onchange={loadAndFit} />

<style>
  .panel-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    padding: 6px 10px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .panel-btn:hover {
    background: var(--hover-bg);
  }
  .panel-btn :global(svg) {
    color: var(--muted-strong);
  }
  .live-anchor {
    position: relative;
    display: inline-block;
  }
  .live-pop {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 10;
    min-width: 220px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--sidebar-bg);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  .live-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }
  .live-row a {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text);
  }
  .live-port {
    color: var(--success);
    font-variant-numeric: tabular-nums;
  }
  .live-port.dead {
    color: var(--danger);
  }
  .live-stop {
    border: 1px solid var(--danger);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--danger);
    cursor: pointer;
    font-size: 11px;
    padding: 2px 8px;
  }
</style>
