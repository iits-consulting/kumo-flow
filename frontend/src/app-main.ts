// Entry for the deployed app page (app.html) — served by deploy.py at "/"
// when the artifact carries a ui spec. Everything is same-origin, and nothing
// here imports the canvas (@xyflow) side of the editor.
import { mount } from "svelte";

import AppStandalone from "./lib/render/AppStandalone.svelte";

import "@fontsource/poppins/latin-400.css";
import "@fontsource/poppins/latin-600.css";
import "./app.css";

// app.css themes via the .dark class the editor toggles; the deployed page
// has no toggle, so follow the OS.
if (matchMedia("(prefers-color-scheme: dark)").matches) document.documentElement.classList.add("dark");
document.documentElement.style.background = "var(--bg)";

export default mount(AppStandalone, {
  target: document.getElementById("app")!,
});
