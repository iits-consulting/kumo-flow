<script lang="ts">
  import Pager from "./Pager.svelte";

  // Paged image preview: absolute (or same-origin) URLs, one per batch item.
  // Hosts size it via --preview-max (node card: default 220px, app page: bigger).
  let { images, alt = "result" }: { images: string[]; alt?: string } = $props();
  let view = $state(0);
  let i = $derived(Math.min(view, images.length - 1));
</script>

<img class="preview" src={images[i]} {alt} />
{#if images.length > 1}
  <Pager count={images.length} bind:index={view} />
{/if}

<style>
  .preview {
    max-width: var(--preview-max, 220px);
    max-height: var(--preview-max, 220px);
    border-radius: 4px;
    display: block;
  }
</style>
