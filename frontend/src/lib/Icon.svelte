<script lang="ts" module>
  // Every icon in the app: one 24×24 stroke path each (subpaths joined into
  // a single `d`), so a new icon is one line here — no icon-font, no dep.
  export const ICONS = {
    play: "M7 5l12 7-12 7z",
    stop: "M6 6h12v12H6z",
    group: "M4 5h16v14H4z",
    ungroup: "M4 5h16v14H4zM9 12h6",
    tasks: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
    arrange: "M3 7h13M13 3l4 4-4 4M21 17H8M11 13l-4 4 4 4",
    save: "M12 3v12M7 10l5 5 5-5M4 19h16",
    load: "M12 15V4M7 8l5-5 5 5M4 19h16",
    deploy:
      "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09zM12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2zM9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5",
    sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
    moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
    "chevrons-left": "M11 17l-5-5 5-5M18 17l-5-5 5-5",
    "chevrons-right": "M13 17l5-5-5-5M6 17l5-5-5-5",
    "chevron-left": "M15 18l-6-6 6-6",
    "chevron-right": "M9 18l6-6-6-6",
    refresh: "M23 4v6h-6M20.49 15a9 9 0 1 1-2.12-9.36L23 10",
    sliders: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
    x: "M18 6L6 18M6 6l12 12",
    plus: "M12 5v14M5 12h14",
    scan: "M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
    droplet: "M12 2.7l5.66 5.66a8 8 0 1 1-11.31 0z",
    search: "M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16M21 21l-4.35-4.35",
    scissors: "M6 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6M20 4L8.12 15.88M14.47 14.48L20 20M8.12 8.12L12 12",
    hash: "M4 9h16M4 15h16M10 3L8 21M16 3l-2 18",
    alert: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01",
    info: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20M12 16v-4M12 8h.01",
    crosshair: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20M22 12h-4M6 12H2M12 6V2M12 22v-4",
    tag: "M20.59 13.41L12 22 2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82zM7 7h.01",
    cpu: "M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3M5 5h14v14H5zM9 9h6v6H9z",
    chat: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
    trash: "M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M10 11v6M14 11v6",
  };
  export type IconName = keyof typeof ICONS;
</script>

<script lang="ts">
  let { name, size = 13 }: { name: IconName; size?: number } = $props();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
><path d={ICONS[name]} /></svg>
