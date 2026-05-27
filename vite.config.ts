import { defineConfig } from "vite";

// Served from https://natekali.github.io/imgupscaler/ — the base must match the repo name
// for asset URLs to resolve correctly on GitHub Pages.
export default defineConfig({
  base: "/imgupscaler/",
  build: {
    target: "es2022",
    sourcemap: false,
  },
});
