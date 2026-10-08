<script lang="ts">
  import { useNodes, useSvelteFlow, type NodeProps } from "@xyflow/svelte";
  import { getContext } from "svelte";
  import { FLOW_CTX, appEligible, APP_INPUT_KINDS, type AppNodeData, type AppSection, type FlowContext, type FlowNodeData } from "./flow";
  import Icon from "./Icon.svelte";

  // The App node: composes an end-user page (title + ordered sections that
  // reference canvas nodes). Frontend-only: its
  // xyflow type is "app", so it never reaches the backend.
  let { id, data }: NodeProps = $props();
  let d = $derived(data as AppNodeData);
  const flow = getContext<FlowContext>(FLOW_CTX);
  const { updateNodeData } = useSvelteFlow();
  const canvasNodes = useNodes();

  // Live canvas node data by id — for section labels and the "missing" state.
  let byId = $derived(
    new Map(canvasNodes.current.filter((n) => n.type === "flow").map((n) => [n.id, n.data as FlowNodeData])),
  );
  // What the picker offers: renderable nodes not yet referenced.
  let eligible = $derived(
    canvasNodes.current.filter(
      (n) =>
        n.type === "flow" &&
        appEligible(n.data as FlowNodeData) &&
        !d.sections.some((s) => s.node === n.id),
    ),
  );

  const set = (patch: Partial<AppNodeData>) => updateNodeData(id, patch);
  const setSection = (i: number, patch: Partial<AppSection>) =>
    set({ sections: d.sections.map((s, j) => (j === i ? { ...s, ...patch } : s)) });
  function move(i: number, dir: -1 | 1) {
    const s = [...d.sections];
    [s[i], s[i + dir]] = [s[i + dir], s[i]];
    set({ sections: s });
  }
  function addSection(e: Event) {
    const sel = e.currentTarget as HTMLSelectElement;
    if (sel.value) set({ sections: [...d.sections, { node: sel.value }] });
    sel.value = "";
  }

  // This node's deployment, matched by the name stored at deploy time.
  let live = $derived(flow.deployments.find((dep) => dep.name === d.deployment && dep.alive));
</script>

<div class="app-node">
  <div class="app-node__title">
    <Icon name="tasks" size={12} />
    <input
      class="title-input nodrag"
      value={d.title}
      oninput={(e) => set({ title: (e.currentTarget as HTMLInputElement).value })}
    />
  </div>
  <div class="app-node__body nodrag">
    {#each d.sections as s, i}
      {@const node = byId.get(s.node)}
      <div class="section">
        <div class="section__head">
          {#if node}
            <span class="section__kind">{node.label}</span>
          {:else}
            <span class="section__missing"><Icon name="alert" size={10} /> missing node</span>
          {/if}
          <span class="section__btns">
            <button onclick={() => move(i, -1)} disabled={i === 0} title="move up" aria-label="move up">↑</button>
            <button onclick={() => move(i, 1)} disabled={i === d.sections.length - 1} title="move down" aria-label="move down">↓</button>
            <button onclick={() => set({ sections: d.sections.filter((_, j) => j !== i) })} title="remove section" aria-label="remove section">
              <Icon name="x" size={10} />
            </button>
          </span>
        </div>
        <input
          class="field"
          placeholder={node ? node.label : "label"}
          value={s.label ?? ""}
          oninput={(e) => setSection(i, { label: (e.currentTarget as HTMLInputElement).value })}
        />
        <input
          class="field"
          placeholder="description (optional)"
          value={s.description ?? ""}
          oninput={(e) => setSection(i, { description: (e.currentTarget as HTMLInputElement).value })}
        />
      </div>
    {/each}
    <select class="add" onchange={addSection}>
      <option value="">+ Add section…</option>
      {#each eligible as n}
        {@const nd = n.data as FlowNodeData}
        <option value={n.id}>{nd.label} ({nd.kind in APP_INPUT_KINDS ? "upload" : "result"})</option>
      {/each}
    </select>
    <div class="actions">
      <button class="action" onclick={() => flow.openApp(id)}><Icon name="play" size={10} /> View</button>
      {#if live}
        <a class="live" href={`http://localhost:${live.port}`} target="_blank">live at :{live.port}</a>
        <button class="action" onclick={() => flow.stopDeployment(live.name)}><Icon name="stop" size={10} /> Stop</button>
      {:else}
        <button class="action" onclick={() => flow.deployApp(id)}><Icon name="deploy" size={10} /> Deploy</button>
      {/if}
    </div>
  </div>
</div>

<style>
  .app-node {
    border: 2px solid #0ea5e9; /* = APP_ITEM.color */
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    width: 230px;
    font-size: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .app-node__title {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #0ea5e9;
    color: #fff;
    padding: 6px 10px;
    font-weight: 600;
  }
  .title-input {
    flex: 1;
    min-width: 0;
    border: none;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    color: #fff;
    font: inherit;
    padding: 2px 6px;
  }
  .title-input:focus {
    outline: 1px solid rgba(255, 255, 255, 0.6);
  }
  .app-node__body {
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    cursor: default;
  }
  .section {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 6px;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
  }
  .section__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }
  .section__kind {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .section__missing {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--danger);
    font-weight: 600;
  }
  .section__btns {
    display: inline-flex;
    gap: 2px;
    flex: none;
  }
  .section__btns button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    padding: 0;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    font-size: 10px;
    line-height: 1;
  }
  .section__btns button:disabled {
    opacity: 0.35;
    cursor: default;
  }
  .field {
    width: 100%;
    box-sizing: border-box;
    padding: 4px 6px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--input-bg);
    color: var(--text);
    font-size: 11px;
  }
  .add {
    width: 100%;
    padding: 5px 6px;
    border: 1px dashed var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--muted-strong);
    font-size: 11px;
    cursor: pointer;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .action {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    flex: 1;
    padding: 5px 8px;
    border: none;
    border-radius: 4px;
    background: var(--accent);
    color: var(--accent-text);
    cursor: pointer;
    font-size: 12px;
  }
  .live {
    flex: 1;
    text-align: center;
    color: var(--success);
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
  }
</style>
