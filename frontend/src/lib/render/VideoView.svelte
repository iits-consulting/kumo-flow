<script lang="ts">
  import Pager from "./Pager.svelte";

  // Native player streaming from /media: only the shown clip is fetched, and
  // only as far as it's played or seeked (Range requests). `clips` are full
  // URLs — the host prefixes its API base.
  let { clips }: { clips: string[] } = $props();
  let view = $state(0);
  let i = $derived(Math.min(view, clips.length - 1));
</script>

<!-- svelte-ignore a11y_media_has_caption -->
<video class="preview nodrag" controls preload="metadata" src={clips[i]}></video>
{#if clips.length > 1}
  <Pager count={clips.length} bind:index={view} label="clip " />
{/if}

<style>
  .preview {
    max-width: var(--preview-max, 220px);
    max-height: var(--preview-max, 220px);
    border-radius: 4px;
    display: block;
  }
</style>
