import { mount } from "svelte";

import App from "./App.svelte";

import "@fontsource/poppins/latin-400.css";
import "@fontsource/poppins/latin-600.css";
import "./app.css";

const app = mount(App, {
  target: document.getElementById("app")!,
});

export default app;
