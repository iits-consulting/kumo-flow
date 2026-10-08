<script lang="ts">
  // File/folder pickers of the given media kind. Filters to matching files
  // and hands the host the raw File[] — what happens next is the host's
  // business (the editor reads/uploads them, the deployed page holds them for
  // its multipart /run). With `thumbs`, chosen images preview inline and the
  // pick is named — the app-page look; node cards render their own summary.
  let {
    media,
    onFiles,
    thumbs = false,
  }: { media: "image" | "video" | "volume"; onFiles: (files: File[]) => void; thumbs?: boolean } = $props();

  // Browsers resolve both the picker's accept="video/*" wildcard and File.type
  // through the OS extension→MIME registry, so real media gets hidden in the
  // dialog or dropped by a MIME filter whenever that mapping is missing (.mov
  // on some Linuxes, .mkv, .tga, .ppm, .heic, every DICOM/NIfTI). One explicit
  // extension list feeds the accept string (pickers match extensions literally)
  // and the post-pick filter; the backend decoder is the real validator.
  // "volume" passes everything: DICOM series files usually have NO extension.
  const EXT: Record<string, string[]> = {
    image: "tif tiff bmp ico tga pbm pgm ppm pnm pfm dds jp2 j2k heic heif avif qoi jxl gif jpg jpeg jfif png webp".split(" "),
    video: "mp4 webm mov avi mkv m4v ts flv wmv 3gp mpg mpeg".split(" "),
  };
  const ACCEPT: Record<string, string> = {
    image: "image/*," + EXT.image.map((e) => `.${e}`).join(","),
    video: "video/*," + EXT.video.map((e) => `.${e}`).join(","),
    volume: ".dcm,.ima,.nii,.gz,application/dicom",
  };

  let urls = $state<string[]>([]);
  let name = $state("");

  function pick(e: Event) {
    const files = Array.from((e.currentTarget as HTMLInputElement).files ?? []).filter(
      (f) => media === "volume" || f.type.startsWith(`${media}/`) || EXT[media].includes(f.name.split(".").pop()?.toLowerCase() ?? ""),
    );
    if (!files.length) return;
    if (thumbs) {
      for (const u of urls) URL.revokeObjectURL(u);
      urls = media === "image" ? files.map((f) => URL.createObjectURL(f)) : [];
      name = files.length === 1 ? files[0].name : `${files.length} ${media}s`;
    }
    onFiles(files);
  }
</script>

<div class="pickers">
  <!-- stopPropagation: on node cards a body click opens the settings popover -->
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
  <label class="file-btn nodrag" onclick={(e) => e.stopPropagation()}>Choose File
    <input type="file" accept={ACCEPT[media]} multiple onchange={pick} hidden />
  </label>
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
  <label class="file-btn nodrag" onclick={(e) => e.stopPropagation()}>Choose Folder
    <input type="file" onchange={pick} hidden {...{ webkitdirectory: true }} />
  </label>
</div>
{#if name}<span class="muted">{name}</span>{/if}
{#if urls.length}
  <div class="thumbs">
    {#each urls as u}<img src={u} alt="chosen file" />{/each}
  </div>
{/if}

<style>
  .pickers {
    display: flex;
    flex-direction: column; /* node card: stacked, like the old Load buttons */
    gap: 6px;
  }
  /* app page (thumbs mode): side by side, capped width */
  :global(.app-page) .pickers {
    flex-direction: row;
  }
  :global(.app-page) .file-btn {
    min-width: 140px;
  }
  .file-btn {
    display: block;
    padding: 5px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    text-align: center;
    cursor: pointer;
  }
  .file-btn:hover {
    background: var(--hover-bg);
  }
  .muted {
    color: var(--muted);
    font-size: 12px;
  }
  .thumbs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .thumbs img {
    width: 72px;
    height: 72px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
  }
</style>
