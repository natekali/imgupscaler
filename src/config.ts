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
export interface AppConfig {
  /** Tier 1 — Modal serverless GPU endpoint (reliable primary). Full https URL of the POST endpoint. */
  modalEndpoint: string;
  /** Tier 2 — Hugging Face ZeroGPU Space id, e.g. "natekali/pid-upscaler" (free accelerator). */
  hfSpace: string;
  /** Tier 3 — in-browser Real-ESRGAN x4 ONNX model (always-available floor). */
  onnxModelUrl: string;
  /** Per-tier wall-clock timeouts (ms) before failing over to the next tier. */
  timeouts: {
    modalMs: number;
    hfSpaceMs: number;
  };
  /** Largest edge (px) we send to a remote backend; bigger uploads are downscaled first. */
  maxRemoteInputEdge: number;
}

export const config: AppConfig = {
  // Filled in after `modal deploy` (Phase 4). Example: "https://natekali--pid-upscaler-upscale.modal.run"
  modalEndpoint: "",

  // Filled in after the HF Space is created (Phase 3). Example: "natekali/pid-upscaler"
  hfSpace: "",

  // Real-ESRGAN general x4 v3 (4.9 MB), vendored in public/models and served same-origin
  // from Pages — no CORS, no third-party uptime dependency. Lazy-loaded only when Tier 3
  // runs. BASE_URL makes this resolve in dev ("/…") and on Pages ("/imgupscaler/…").
  onnxModelUrl: `${import.meta.env.BASE_URL}models/realesr-general-x4v3.onnx`,

  timeouts: {
    modalMs: 90_000,
    hfSpaceMs: 90_000,
  },

  maxRemoteInputEdge: 1536,
};
