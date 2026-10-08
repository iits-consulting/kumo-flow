<script lang="ts">
  import { tick } from "svelte";
  import { API, type PaletteItem } from "./flow";
  import type { ChatTurn, Plan } from "./chat";
  import Icon from "./Icon.svelte";

  let {
    byKind,
    getGraph,
    onInsert,
  }: {
    byKind: Record<string, PaletteItem>;
    getGraph: () => unknown; // toBackendPayload of the live canvas
    onInsert: (plan: Plan, outputHandling: ChatTurn["output_handling"]) => void;
  } = $props();

  interface Msg {
    role: "user" | "assistant";
    content: string;
    plan?: Plan | null; // kept on assistant messages so cards survive reload
    output_handling?: ChatTurn["output_handling"]; // what the plan's insert needs
    inserted?: boolean;
  }

  // Mirrors the palette: collapsed by default, state persisted.
  let open = $state(localStorage.getItem("chat") === "open");
  $effect(() => localStorage.setItem("chat", open ? "open" : "closed"));

  // Frontend owns all chat state; brief = the latest non-null fields from
  // responses, echoed back with every request.
  let messages = $state<Msg[]>([]);
  let brief = $state<{ input_kind: string | null; output_handling: string | null }>({
    input_kind: null,
    output_handling: null,
  });
  try {
    const saved = JSON.parse(localStorage.getItem("chat-log") ?? "");
    messages = saved.messages ?? [];
    if (saved.brief) brief = saved.brief;
  } catch {} // corrupt/absent history — start empty
  $effect(() => localStorage.setItem("chat-log", JSON.stringify({ messages, brief })));

  let input = $state("");
  let pending = $state(false);
  let error = $state("");
  let inputEl = $state<HTMLTextAreaElement | null>(null);
  let logEl = $state<HTMLDivElement | null>(null);
  let ctrl: AbortController | null = null;

  // The start page's "describe what you want to build" entry.
  export function show() {
    open = true;
    tick().then(() => inputEl?.focus());
  }

  $effect(() => {
    void messages.length, pending; // new bubble or thinking indicator -> follow it
    tick().then(() => logEl?.scrollTo({ top: logEl.scrollHeight }));
  });

  function submit() {
    const text = input.trim();
    if (!text || pending) return;
    input = "";
    messages = [...messages, { role: "user", content: text }];
    send();
  }

  // Sends the current history (a failed attempt retries by calling this again —
  // the user message is already in `messages`). Assistant messages that carried
  // a plan get its JSON appended: the model only ever sees role+content, and
  // without this it can't refine a plan it proposed ("add a resize before the
  // blur") — the prose reply alone doesn't say what it built.
  async function send() {
    error = "";
    pending = true;
    ctrl = new AbortController();
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          messages: messages.slice(-20).map(({ role, content, plan }) => ({
            role,
            content: plan ? `${content}\n\n[my proposed plan: ${JSON.stringify(plan)}]` : content,
          })),
          graph: getGraph(),
          brief,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const turn: ChatTurn = await res.json();
      messages = [
        ...messages,
        { role: "assistant", content: turn.reply, plan: turn.plan, output_handling: turn.output_handling },
      ];
      brief = {
        input_kind: turn.input_kind ?? brief.input_kind,
        output_handling: turn.output_handling ?? brief.output_handling,
      };
    } catch (e) {
      // aborted = the user pressed Stop; not an error, no retry bubble
      if ((e as Error).name !== "AbortError") error = `${e}`;
    } finally {
      pending = false;
      ctrl = null;
    }
  }

  function insert(m: Msg) {
    if (!m.plan || m.inserted) return;
    onInsert(m.plan, m.output_handling ?? null);
    m.inserted = true; // deep $state proxy — persists via the effect above
  }

  function clear() {
    messages = [];
    brief = { input_kind: null, output_handling: null };
    error = "";
  }

  const stepLabels = (plan: Plan) => plan.steps.map((s) => byKind[s.kind]?.label ?? s.kind).join(" → ");
</script>

{#if open}
  <aside class="chat">
    <div class="chat__head">
      <h2>Assistant</h2>
      <div class="chat__head-actions">
        {#if messages.length}
          <button class="icon-btn" onclick={clear} title="Clear conversation" aria-label="Clear conversation">
            <Icon name="trash" size={14} />
          </button>
        {/if}
        <button class="icon-btn" onclick={() => (open = false)} title="Hide assistant" aria-label="Hide assistant">
          <Icon name="chevrons-right" size={14} />
        </button>
      </div>
    </div>

    <div class="chat__log" bind:this={logEl}>
      {#if !messages.length}
        <p class="chat__hint">
          Describe what you want to build — e.g. “blur the background of my images” — and I'll propose a pipeline you
          can add to the canvas. I can also explain the current graph.
        </p>
      {/if}
      {#each messages as m}
        <div class="chat__msg chat__msg--{m.role}">
          <div class="chat__bubble">{m.content}</div>
          {#if m.plan}
            <div class="chat__plan">
              <div class="chat__plan-steps">{stepLabels(m.plan)}</div>
              <button class="chat__plan-add" disabled={m.inserted} onclick={() => insert(m)}>
                {m.inserted ? "Added ✓" : "Add to canvas"}
              </button>
            </div>
          {/if}
        </div>
      {/each}
      {#if pending}
        <div class="chat__msg chat__msg--assistant chat__msg--pending">
          <div class="chat__bubble chat__bubble--thinking">Thinking…</div>
          <!-- frees the UI only — the backend finishes its turn server-side
               and the response is discarded; local models can grind for minutes -->
          <button class="chat__plan-add" onclick={() => ctrl?.abort()}>Stop</button>
        </div>
      {/if}
      {#if error}
        <div class="chat__error">
          <p>Couldn't reach the assistant ({error}). Is Ollama running? <code>ollama serve</code>, and is the
            configured model pulled?</p>
          <button class="chat__plan-add" onclick={send} disabled={pending}>Retry</button>
        </div>
      {/if}
    </div>

    <form
      class="chat__inputrow"
      onsubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        class="chat__input"
        rows="2"
        placeholder="Describe what you want to build…"
        bind:value={input}
        bind:this={inputEl}
        disabled={pending}
        onkeydown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      ></textarea>
      <button class="chat__send" type="submit" disabled={pending || !input.trim()} aria-label="Send">
        <Icon name="play" size={12} />
      </button>
    </form>
  </aside>
{:else}
  <button class="chat-opener" onclick={show} title="Open the assistant" aria-label="Open the assistant">
    <Icon name="chat" size={16} />
  </button>
{/if}

<style>
  /* Mirrors the left palette's look (App.svelte .palette) on the right edge. */
  .chat {
    width: 300px;
    flex: 0 0 300px;
    display: flex;
    flex-direction: column;
    border-left: 1px solid var(--border);
    background: var(--sidebar-bg);
    box-sizing: border-box;
  }
  .chat__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 12px 8px;
  }
  .chat h2 {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-strong);
    margin: 0;
  }
  .chat__head-actions {
    display: flex;
    gap: 4px;
  }
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    line-height: 1;
    padding: 5px;
  }
  .icon-btn:hover {
    background: var(--hover-bg);
    color: var(--text);
  }
  .chat-opener {
    position: absolute;
    top: 50%;
    right: 0;
    transform: translateY(-50%);
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-strong);
    border-right: none;
    border-radius: 8px 0 0 8px;
    background: var(--card-bg);
    color: var(--muted-strong);
    cursor: pointer;
    padding: 10px 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }
  .chat-opener:hover {
    background: var(--hover-bg);
    color: var(--text);
  }
  .chat__log {
    flex: 1;
    overflow-y: auto;
    padding: 4px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .chat__hint {
    color: var(--muted);
    font-size: 12px;
  }
  .chat__msg {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    max-width: 100%;
  }
  .chat__msg--user {
    align-items: flex-end;
  }
  .chat__bubble {
    padding: 7px 10px;
    border-radius: 10px;
    background: var(--card-bg);
    border: 1px solid var(--border-strong);
    font-size: 13px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-width: 92%;
    box-sizing: border-box;
  }
  .chat__msg--user .chat__bubble {
    background: var(--accent);
    color: var(--accent-text);
    border-color: var(--accent);
  }
  .chat__bubble--thinking {
    color: var(--muted);
    animation: chat-pulse 1.2s ease-in-out infinite;
  }
  .chat__msg--pending {
    flex-direction: row;
    align-items: center;
    gap: 6px;
  }
  @keyframes chat-pulse {
    50% {
      opacity: 0.4;
    }
  }
  .chat__plan {
    margin-top: 6px;
    padding: 8px 10px;
    border: 1px solid var(--border-strong);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    background: var(--card-bg);
    max-width: 92%;
    box-sizing: border-box;
  }
  .chat__plan-steps {
    font-size: 12px;
    color: var(--text);
    margin-bottom: 8px;
  }
  .chat__plan-add {
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    padding: 5px 10px;
  }
  .chat__plan-add:hover:not(:disabled) {
    background: var(--hover-bg);
  }
  .chat__plan-add:disabled {
    color: var(--success);
    cursor: default;
  }
  .chat__error {
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    color: var(--danger);
  }
  .chat__error p {
    margin: 0 0 6px;
  }
  .chat__inputrow {
    display: flex;
    gap: 6px;
    padding: 8px 12px 12px;
    border-top: 1px solid var(--border);
  }
  .chat__input {
    flex: 1;
    resize: none;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--input-bg);
    color: var(--text);
    font-size: 13px;
    font-family: inherit;
    padding: 6px 8px;
    box-sizing: border-box;
  }
  .chat__send {
    align-self: flex-end;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    padding: 8px;
  }
  .chat__send:disabled {
    color: var(--muted);
    cursor: default;
  }
  .chat__send:hover:not(:disabled) {
    background: var(--hover-bg);
  }
</style>
