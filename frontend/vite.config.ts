import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    tailwindcss(),
    svelte({
      emitCss: false,
    }),
  ],
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
  build: {
    rollupOptions: {
      // two pages: the editor (index) and the deployed app page (app),
      // which deploy.py serves as static files when an artifact has a ui
      input: {
        index: fileURLToPath(new URL("index.html", import.meta.url)),
        app: fileURLToPath(new URL("app.html", import.meta.url)),
      },
    },
  },
});
