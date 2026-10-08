<script lang="ts">
  import type { ExportResult } from "./types";

  let { download }: { download: ExportResult } = $props();
</script>

<!-- plain <a download>: the content-addressed hash name on disk becomes
     a readable filename in the user's Downloads folder -->
<a class="file-btn nodrag" href={download.url} download="kumoflow-export.zip">Download zip</a>
<span class="muted">{download.summary}</span>

<style>
  .file-btn {
    display: block;
    padding: 5px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--card-bg);
    color: var(--text);
    text-align: center;
    cursor: pointer;
    text-decoration: none;
  }
  .file-btn:hover {
    background: var(--hover-bg);
  }
  .muted {
    color: var(--muted);
    font-size: 12px;
  }
</style>
