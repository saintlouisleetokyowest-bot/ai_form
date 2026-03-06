import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@ai-form/core": path.resolve(__dirname, "../..", "packages/core/src"),
      "@ai-form/models": path.resolve(__dirname, "../..", "packages/models/src"),
      "@ai-form/config": path.resolve(__dirname, "../..", "config"),
    },
  },
  server: {
    port: 5173,
  },
});
