import type { Edge, Node } from "@xyflow/svelte";
import { createNode, makeGroup, type PaletteItem } from "./flow";
import type { IconName } from "./Icon.svelte";

// A recipe is a small pre-wired subgraph for a user goal ("blur the
// background") rather than a single primitive node. dx/dy lay out each step
// relative to the recipe's drop position; wiring connects them by port name.
export interface RecipeStep {
  kind: string;
  dx: number;
  dy: number;
  config?: Record<string, unknown>; // overrides the node's default config values
}

export interface RecipeWire {
  from: number; // index into steps
  port: string;
  to: number;
  targetPort: string;
}

// Visually bundles a cluster of steps (indices into `steps`) into one labeled
// subflow box on instantiation — e.g. a prompt wired straight into its model.
export interface RecipeGroup {
  label: string;
  steps: number[];
}

export interface Recipe {
  name: string;
  description: string;
  icon: IconName;
  steps: RecipeStep[];
  wiring: RecipeWire[];
  groups?: RecipeGroup[];
}

export const RECIPES: Recipe[] = [
  {
    name: "Segment Subject",
    icon: "scan",
    description: "Find and outline something described in text",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "segment", dx: 280, dy: 260 },
      { kind: "view", dx: 560, dy: 260 },
      // Every recipe ends with a guide note: last in `steps` so the group
      // indices above stay put, and absent from `wiring` — a note has no ports.
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Segment Subject — outlines whatever you name in words.\n\n" +
            "1. Load Image: one file, or a folder for a whole batch.\n" +
            '2. Text Prompt: the concepts to find, comma separated ("person, truck").\n' +
            "3. ▶ Run on View Image.\n\n" +
            "SAM3 returns one mask + box per instance; View Image draws them over the original.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 0, port: "image", to: 3, targetPort: "image" },
      { from: 2, port: "masks", to: 3, targetPort: "masks" },
      { from: 2, port: "boxes", to: 3, targetPort: "boxes" },
    ],
    groups: [{ label: "Segment", steps: [1, 2] }],
  },
  {
    name: "Blur Background",
    icon: "droplet",
    description: "Segment the subject and blur everything else",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "segment", dx: 280, dy: 260 },
      { kind: "blur", dx: 560, dy: 260 },
      { kind: "view", dx: 840, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Blur Background — blurs the pixels SAM3 matched.\n\n" +
            "1. Load Image: file or folder.\n" +
            '2. Text Prompt: what to blur ("face, license plate").\n' +
            "3. ▶ Run on View Image.\n\n" +
            "Blur only touches what is inside the masks, so prompt the part you want smeared, " +
            "not the part you want kept. Kernel size is Blur's setting.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 0, port: "image", to: 3, targetPort: "image" },
      { from: 2, port: "masks", to: 3, targetPort: "masks" },
      { from: 3, port: "image", to: 4, targetPort: "image" },
    ],
    groups: [{ label: "Segment", steps: [1, 2] }],
  },
  {
    name: "Identify then Classify",
    icon: "search",
    description: "Find an object by text, crop to it, then classify the crop with your own labels",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "segment", dx: 280, dy: 260 },
      { kind: "crop", dx: 560, dy: 260 },
      { kind: "text_prompt", dx: 840, dy: 0 },
      { kind: "classify", dx: 840, dy: 260 },
      { kind: "view_classification", dx: 1120, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Identify then Classify — SAM3 finds and crops the object, SigLIP 2 labels the crop.\n\n" +
            "1. Load Image: file or folder.\n" +
            '2. Identify › Text Prompt: what to find ("dog").\n' +
            '3. Classify › Text Prompt: your candidate labels ("sitting, standing, lying").\n' +
            "4. ▶ Run on View Classification.\n\n" +
            "Every box becomes its own item in the batch, so you get one score set per detection, not per image.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 0, port: "image", to: 3, targetPort: "image" },
      { from: 2, port: "boxes", to: 3, targetPort: "boxes" },
      { from: 3, port: "image", to: 5, targetPort: "image" },
      { from: 4, port: "prompts", to: 5, targetPort: "prompts" },
      { from: 5, port: "classification", to: 6, targetPort: "classification" },
    ],
    groups: [
      { label: "Identify", steps: [1, 2] },
      { label: "Classify", steps: [4, 5] },
    ],
  },
  {
    name: "Remove Background",
    icon: "scissors",
    description: "Segment the subject and white out everything around it",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "segment", dx: 280, dy: 260 },
      {
        kind: "custom",
        dx: 560,
        dy: 260,
        // ponytail: no dedicated "apply mask" node — a preset Custom Code node
        // does the 4-line apply; promote to a real node if a second recipe needs it.
        config: {
          input_ports: ["image", "masks"],
          output_ports: ["image"],
          code:
            "def run(image, masks):\n" +
            "    # keep pixels inside any instance mask, white out the rest\n" +
            "    out = []\n" +
            "    for im, per in zip(image, masks):\n" +
            "        res = im.clone()\n" +
            "        res[:, ~(per > 0).any(0)] = 255\n" +
            "        out.append(res)\n" +
            '    return {"image": out}\n',
        },
      },
      { kind: "view", dx: 840, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Remove Background — keeps the prompted subject, whites out everything around it.\n\n" +
            "1. Load Image: file or folder.\n" +
            '2. Text Prompt: the subject to keep ("person").\n' +
            "3. ▶ Run on View Image.\n\n" +
            "The Custom Code node does the masking — open it to change white (255) to another value.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 0, port: "image", to: 3, targetPort: "image" },
      { from: 2, port: "masks", to: 3, targetPort: "masks" },
      { from: 3, port: "image", to: 4, targetPort: "image" },
    ],
    groups: [{ label: "Segment", steps: [1, 2] }],
  },
  {
    name: "Count Objects",
    icon: "hash",
    description: "Count everything matching a text prompt, per image and across the batch",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "segment", dx: 280, dy: 260 },
      { kind: "count", dx: 560, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Count Objects — how many matches per image, plus a batch total.\n\n" +
            "1. Load Image: file or folder.\n" +
            '2. Text Prompt: the concepts to count ("car, person").\n' +
            "3. ▶ Run on Count Objects.\n\n" +
            'One column per concept. Overlapping phrases ("car, vehicle") count the same object twice.',
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 2, port: "boxes", to: 3, targetPort: "boxes" },
      // per-instance concept names: one count column per phrase of the prompt
      { from: 2, port: "labels", to: 3, targetPort: "labels" },
    ],
    groups: [{ label: "Detect", steps: [1, 2] }],
  },
  {
    name: "Find Defects",
    icon: "alert",
    description:
      "Train on good parts only (pick a folder with a good/ subfolder): just the flagged parts, most unusual first, defect region marked",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "patchcore", dx: 280, dy: 0, config: { normal_label: "good" } },
      // 0.5 is where PatchCore puts its calibrated line ("as unusual as a good
      // part gets"), so this keeps the flagged parts and drops the rest instead
      // of paging through a batch that is mostly clean. label_name stays empty =
      // the first classifier label, "defect" — that port carries PatchCore's own
      // labels, never the good/ folder's name.
      { kind: "filter_class", dx: 560, dy: 0, config: { threshold: 0.5 } },
      { kind: "view", dx: 840, dy: 0 },
      { kind: "view_classification", dx: 560, dy: 320 },
      {
        kind: "note",
        dx: 0,
        dy: 320,
        config: {
          text:
            "Find Defects — PatchCore learns from good parts only, then flags the odd ones.\n\n" +
            "1. Load Image: pick the dataset folder. It needs a good/ subfolder — those parts are the reference.\n" +
            "2. ▶ Run on View Image: the flagged parts, most unusual first, defect region marked.\n" +
            "3. ▶ Run on View Classification for the scores. 50% = as unusual as a good part still gets.\n\n" +
            "Flagging too much? Raise Find Defects' tolerance, or Filter by Class's threshold.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 1, targetPort: "image" },
      // which images are the good ones: the folder labels decide
      { from: 0, port: "labels", to: 1, targetPort: "labels" },
      // most unusual first; the defect regions ride along so each overlay stays on its own image
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "classification", to: 2, targetPort: "classification" },
      { from: 1, port: "masks", to: 2, targetPort: "masks" },
      { from: 1, port: "boxes", to: 2, targetPort: "boxes" },
      { from: 2, port: "image", to: 3, targetPort: "image" },
      { from: 2, port: "masks", to: 3, targetPort: "masks" },
      { from: 2, port: "boxes", to: 3, targetPort: "boxes" },
      // second output: the score per part, 50% being the "as bad as a good part gets" line
      { from: 1, port: "classification", to: 4, targetPort: "classification" },
    ],
  },
  {
    name: "Track & Count Objects",
    icon: "crosshair",
    description: "Follow everything matching a text prompt through a video, watch the tracks, count each object once",
    steps: [
      { kind: "load_video", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 320 },
      { kind: "segment", dx: 280, dy: 320 },
      { kind: "track", dx: 560, dy: 320 },
      { kind: "view_video", dx: 840, dy: 0 },
      { kind: "count", dx: 840, dy: 380 },
      {
        kind: "note",
        dx: 0,
        dy: 580,
        config: {
          text:
            "Track & Count — follows the prompted objects through the footage and counts each one once.\n\n" +
            "1. Load Video: file or folder of clips. Lower fps decodes fewer frames.\n" +
            '2. Text Prompt: what to follow ("person, bicycle").\n' +
            "3. ▶ Run on View Video (tracks drawn: one color + id per object), and on Count Objects (one row per clip).\n\n" +
            "Track Objects matches boxes by overlap between frames, so fast motion at a low fps splits a track " +
            "in two — lower its iou_threshold if ids keep changing.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "video", to: 2, targetPort: "video" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      // per-frame detections + their concept names -> identity across frames
      { from: 2, port: "boxes", to: 3, targetPort: "boxes" },
      { from: 2, port: "labels", to: 3, targetPort: "labels" },
      // the tracks, drawn on the footage: one color and one number per object
      { from: 0, port: "video", to: 4, targetPort: "video" },
      { from: 2, port: "masks", to: 4, targetPort: "masks" },
      { from: 2, port: "boxes", to: 4, targetPort: "boxes" },
      { from: 3, port: "ids", to: 4, targetPort: "ids" },
      // and counted: one row per clip, each track counted once
      { from: 2, port: "boxes", to: 5, targetPort: "boxes" },
      { from: 2, port: "labels", to: 5, targetPort: "labels" },
      { from: 3, port: "ids", to: 5, targetPort: "ids" },
    ],
    groups: [{ label: "Detect", steps: [1, 2] }],
  },
  {
    name: "Classify Objects",
    icon: "tag",
    description: "Score an image against candidate text labels",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "text_prompt", dx: 0, dy: 260 },
      { kind: "classify", dx: 280, dy: 260 },
      { kind: "view_classification", dx: 560, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 520,
        config: {
          text:
            "Classify Objects — zero-shot: scores each image against labels you type. No training, no boxes.\n\n" +
            "1. Load Image: file or folder.\n" +
            '2. Text Prompt: your candidate labels ("cat, dog, neither").\n' +
            "3. ▶ Run on View Classification.\n\n" +
            'Scores only compare the labels you give, so add a catch-all label if "none of these" is a real answer. ' +
            "To label an object instead of the whole frame, use the Identify then Classify task.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 2, targetPort: "image" },
      { from: 1, port: "prompts", to: 2, targetPort: "prompts" },
      { from: 2, port: "classification", to: 3, targetPort: "classification" },
    ],
    groups: [{ label: "Classify", steps: [1, 2] }],
  },
  {
    name: "Train Classifier",
    icon: "cpu",
    description: "Embed images, fit a classifier on your labels (e.g. from a class-per-folder pick), map the batch",
    steps: [
      { kind: "load", dx: 0, dy: 0 },
      { kind: "embed", dx: 280, dy: 0 },
      { kind: "logreg", dx: 560, dy: 0 },
      { kind: "view_classification", dx: 840, dy: 0 },
      { kind: "umap", dx: 560, dy: 260 },
      { kind: "show_embeddings", dx: 840, dy: 260 },
      {
        kind: "note",
        dx: 0,
        dy: 260,
        config: {
          text:
            "Train Classifier — DINOv2 embeds every image, logistic regression learns your labels and predicts the batch.\n\n" +
            "1. Load Image: pick a folder with one subfolder per class; the subfolder names become the labels.\n" +
            "2. ▶ Run on View Classification for the predictions.\n" +
            "3. ▶ Run on View Embeddings for the same images as a 2-D map — hover a point to see its image.\n\n" +
            "No labeled folders? Type the labels into Train Classifier instead, one line per image, in batch order.",
        },
      },
    ],
    wiring: [
      { from: 0, port: "image", to: 1, targetPort: "image" },
      { from: 1, port: "embedding", to: 2, targetPort: "embedding" },
      { from: 0, port: "labels", to: 2, targetPort: "labels" },
      { from: 2, port: "classification", to: 3, targetPort: "classification" },
      // second output: the same embeddings reduced to 2-D, with hover thumbnails
      { from: 1, port: "embedding", to: 4, targetPort: "embedding" },
      { from: 4, port: "embedding", to: 5, targetPort: "embedding" },
      { from: 0, port: "image", to: 5, targetPort: "image" },
    ],
  },
];

// kind -> kind -> number of recipe wires connecting them (any port) — the
// ranking data behind ghost suggestions and menu ordering (App.svelte).
export const AFFINITY: Record<string, Record<string, number>> = {};
for (const r of RECIPES)
  for (const w of r.wiring) {
    const a = r.steps[w.from].kind, b = r.steps[w.to].kind;
    (AFFINITY[a] ??= {})[b] = (AFFINITY[a][b] ?? 0) + 1;
  }

export function instantiateRecipe(
  recipe: Recipe,
  byKind: Record<string, PaletteItem>,
  origin: { x: number; y: number },
): { nodes: Node[]; edges: Edge[] } {
  const steps = recipe.steps.map((s) => {
    const n = createNode(byKind[s.kind], { x: origin.x + s.dx, y: origin.y + s.dy });
    Object.assign(n.data.config as Record<string, unknown>, s.config);
    return n;
  });
  const edges = recipe.wiring.map((w) => ({
    id: `${steps[w.from].id}-${steps[w.to].id}-${w.port}`,
    source: steps[w.from].id,
    target: steps[w.to].id,
    sourceHandle: w.port,
    targetHandle: w.targetPort,
  }));

  let nodes: Node[] = steps;
  for (const g of recipe.groups ?? []) {
    const ids = new Set(g.steps.map((i) => steps[i].id));
    const subset = nodes.filter((n) => ids.has(n.id));
    const { group, children } = makeGroup(subset, g.label);
    nodes = [...nodes.filter((n) => !ids.has(n.id)), group, ...children];
  }
  return { nodes, edges };
}
