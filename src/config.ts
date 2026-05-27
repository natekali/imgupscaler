/**
 * Backend configuration — PUBLIC, NON-SECRET values only.
 *
 * These URLs are safe to ship in a public static bundle: they are endpoints, not
 * credentials. No API token, HF token, or GitHub PAT ever belongs in this file or any
 * other file under src/ — a GitHub Pages site is world-readable. Backend auth lives
 * server-side (inside the Modal function / HF Space), never in the browser.
 *
 * Tiers the orchestrator tries in order. Leave a value empty ("") to disable that tier;
 * the orchestrator skips unconfigured tiers and falls through to the in-browser engine,
 * so the site is fully functional even before any GPU backend is deployed.
 */
export type Quality = "fast" | "high";

export interface OnnxModel {
  url: string;
  /** Engine label shown to the user (and on the result's "After" tag). */
  label: string;
}

export interface AppConfig {
  /** Tier 1 — Modal serverless GPU endpoint (reliable primary). Full https URL of the POST endpoint. */
  modalEndpoint: string;
  /** Tier 2 — Hugging Face ZeroGPU Space id, e.g. "natekali/pid-upscaler" (free accelerator). */
  hfSpace: string;
  /** Tier 3 — in-browser Real-ESRGAN models, by quality. Vendored same-origin (no CORS). */
  onnxModels: Record<Quality, OnnxModel>;
  defaultQuality: Quality;
  /** Longest input edge (px) before 4× upscaling, per quality. Bounds compute + output size. */
  inputCap: Record<Quality, number>;
  /** Per-tier wall-clock timeouts (ms) before failing over to the next tier. */
  timeouts: { modalMs: number; hfSpaceMs: number };
}

const base = import.meta.env.BASE_URL; // "/" in dev, "/imgupscaler/" on Pages

export const config: AppConfig = {
  // Filled in after `modal deploy` (optional). Example: "https://natekali--pid-upscaler-web.modal.run/upscale"
  modalEndpoint: "",
  // Filled in after the HF Space is created (optional). Example: "natekali/pid-upscaler"
  hfSpace: "",

  onnxModels: {
    // Tiny SRVGG compact model — instant, great default.
    fast: { url: `${base}models/realesr-general-x4v3.onnx`, label: "Real-ESRGAN" },
    // Full RRDBNet x4plus — higher detail, heavier (best with WebGPU).
    high: { url: `${base}models/realesrgan-x4plus.onnx`, label: "Real-ESRGAN HD" },
  },
  defaultQuality: "fast",

  // High mode runs a 23-block network, so cap its input smaller to stay responsive on WASM.
  inputCap: { fast: 1536, high: 832 },

  timeouts: { modalMs: 90_000, hfSpaceMs: 90_000 },
};
