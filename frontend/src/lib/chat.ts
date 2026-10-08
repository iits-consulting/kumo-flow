import type { Edge, Node } from "@xyflow/svelte";
import {
  APP_ITEM,
  appEligible,
  createNode,
  layoutNodes,
  type AppNodeData,
  type FlowNodeData,
  type PaletteItem,
} from "./flow";

// Mirrors backend/chat.py's Plan: steps + wiring only, no coordinates — the
// frontend owns layout (layoutNodes below).
export interface PlanStep {
  kind: string;
  config?: Record<string, unknown>;
}

export interface PlanWire {
  from: number; // index into steps
  port: string;
  to: number;
  target_port: string;
}

export interface Plan {
  steps: PlanStep[];
  wiring: PlanWire[];
}

// One assistant turn as POST /chat returns it.
export interface ChatTurn {
  reply: string;
  input_kind: "image" | "video" | null;
  output_handling: "view" | "export" | "app" | null;
  plan: Plan | null;
}

// A validated plan -> ready-to-append nodes/edges (the chat card's "Add to
// canvas"): instantiate each step with its config overrides (same as
// instantiateRecipe), wire by port name, lay out with layoutNodes, then
// translate the subgraph so its top-left lands at `origin`. With `withApp`
// (output_handling === "app"), an App node lands beside the subgraph with a
// section per eligible node (the load and view/display nodes).
export function insertPlan(
  plan: Plan,
  byKind: Record<string, PaletteItem>,
  origin: { x: number; y: number },
  withApp: boolean,
): { nodes: Node[]; edges: Edge[] } {
  const steps = plan.steps.map((s) => {
    const n = createNode(byKind[s.kind], origin);
    Object.assign(n.data.config as Record<string, unknown>, s.config);
    return n;
  });
  const edges = plan.wiring.map((w) => ({
    id: `${steps[w.from].id}-${steps[w.to].id}-${w.port}`,
    source: steps[w.from].id,
    target: steps[w.to].id,
    sourceHandle: w.port,
    targetHandle: w.target_port,
  }));
  let nodes = layoutNodes(steps, edges); // positions centered around (0,0)
  const dx = origin.x - Math.min(...nodes.map((n) => n.position.x));
  const dy = origin.y - Math.min(...nodes.map((n) => n.position.y));
  nodes = nodes.map((n) => ({ ...n, position: { x: n.position.x + dx, y: n.position.y + dy } }));
  if (withApp) {
    const app = createNode(APP_ITEM, { x: Math.max(...nodes.map((n) => n.position.x)) + 280, y: origin.y });
    (app.data as AppNodeData).sections = nodes
      .filter((n) => appEligible(n.data as FlowNodeData))
      .map((n) => ({ node: n.id }));
    nodes = [...nodes, app];
  }
  return { nodes, edges };
}
