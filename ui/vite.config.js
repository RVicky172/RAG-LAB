/**
 * Vite settings for the React UI (Doc 5).
 *
 * What: Vite is the dev server and build tool. `npm run dev` serves the UI at
 * http://localhost:5173 and reloads it on every save; `npm run build` writes
 * the finished page to ui/dist/, which FastAPI then serves.
 *
 * Why the proxy: in development the UI (port 5173) and the API (port 8000) are
 * two servers. The proxy forwards every /api request to FastAPI, so the code
 * can always call "/api/..." - the same URL it uses when FastAPI serves the
 * built UI itself. No CORS setup needed.
 */

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
