// Plain-data result types shared by the canvas nodes (FlowNode) and the app
// page (AppView). This module must stay free of @xyflow imports — the deployed
// app bundle (app.html) pulls it in and must not drag the canvas library along.

// One per batch image: candidate labels and their (independent, sigmoid) probabilities.
export interface ClassResult {
  labels: string[];
  scores: number[];
}

// Count Objects: one column per concept SAM3 was prompted with (or a single
// "objects" column), a row of counts per batch item — an image, or a clip when
// tracking ids are wired in — and the batch total.
export interface CountResult {
  concepts: string[];
  per_image: number[][];
  total: number[];
}

// View Volume: one raw uint8 grayscale volume per clip (url + [T,H,W] dims),
// optionally a same-dims uint8 label volume (`seg`, voxel = instance index + 1,
// 0 = background) with one name/color per instance index.
export interface VolumeResult {
  url: string;
  dims: number[];
  seg?: string;
  names?: (string | null)[];
  colors?: (number[] | null)[];
}

// Export: the built zip's absolute /media URL and a "14 files, 3.2 MB" line.
export interface ExportResult {
  url: string;
  summary: string;
}

// One display node's shaped result, normalized for the shared renderers.
// `images` doubles as scatter hover thumbnails on show_embeddings nodes.
export interface NodeResult {
  images?: string[];
  classification?: ClassResult[];
  counts?: CountResult;
  download?: ExportResult;
  points?: number[][];
  videos?: string[];
  error?: string;
}

// The backend's per-target /run JSON -> NodeResult, media URLs prefixed with
// `api` ("" on the deployed page, where everything is same-origin).
export function toNodeResult(r: Record<string, any> | undefined, api: string): NodeResult {
  if (!r || r.cancelled) return {};
  if (r.classification) return { classification: r.classification };
  if (r.counts) return { counts: r.counts };
  if (r.download) return { download: { url: api + r.download, summary: r.summary ?? "" } };
  if (r.points) return { points: r.points, images: r.thumbnails };
  if (r.images) return { images: r.images.map((u: string) => api + u) };
  if (r.videos) return { videos: r.videos.map((u: string) => api + u) };
  return { error: r.error ?? "unknown error" };
}
