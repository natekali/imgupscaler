/**
 * Backend configuration, PUBLIC, NON-SECRET values only.
 *
 * These URLs are safe to ship in a public static bundle: they are endpoints, not
 * credentials. No API token, HF token, or GitHub PAT ever belongs in this file or any
 * other file under src/, a GitHub Pages site is world-readable. Backend auth lives
 * server-side (inside the Modal function / HF Space), never in the browser.
 */

/** What the user can pick. fast/high run locally in the browser; pid runs NVIDIA PiD on a GPU. */
export type Engine = "fast" | "high" | "pid";
export type BrowserQuality = "fast" | "high";

export interface OnnxModel {
  url: string;
  /** Engine label shown to the user (and on the result's "After" tag). */
  label: string;
}

export interface AppConfig {
  /** NVIDIA PiD on Modal: base URL of the deployed asgi app (serves /upscale and /health). */
  modalBase: string;
  /** NVIDIA PiD on a Hugging Face ZeroGPU Space id, e.g. "0ximg/pid-upscaler" (secondary). */
  hfSpace: string;
  /** In-browser Real-ESRGAN models, by quality. Vendored same-origin (no CORS). */
  onnxModels: Record<BrowserQuality, OnnxModel>;
  /** Longest input edge (px) before 4× upscaling, per browser quality. Bounds compute + output. */
  inputCap: Record<BrowserQuality, number>;
  /** Wall-clock timeouts (ms). PiD is a cold GPU + diffusion, so it gets a long budget. */
  timeouts: { pidMs: number; healthMs: number };
}

const base = import.meta.env.BASE_URL; // "/" in dev, "/imgupscaler/" on Pages

export const config: AppConfig = {
  // NVIDIA PiD on Modal (deployed as a Modal Cls; model is loaded once per container).
  // The asgi base; provider appends /upscale and /health.
  modalBase: "https://guykalikey--upscaler-ai-pid-pidservice-web.modal.run",
  // Optional secondary PiD backend (HF ZeroGPU Space id).
  hfSpace: "",

  onnxModels: {
    // Tiny SRVGG compact model, instant, great default.
    fast: { url: `${base}models/realesr-general-x4v3.onnx`, label: "Real-ESRGAN" },
    // Full RRDBNet x4plus, higher detail, heavier (best with WebGPU).
    high: { url: `${base}models/realesrgan-x4plus.onnx`, label: "Real-ESRGAN HD" },
  },

  // High mode runs a 23-block network, so cap its input smaller to stay responsive on WASM.
  inputCap: { fast: 1536, high: 832 },

  timeouts: { pidMs: 240_000, healthMs: 5_000 },
};

/** Whether any NVIDIA PiD backend is wired up at all. */
export function pidConfigured(): boolean {
  return config.modalBase.trim().length > 0 || config.hfSpace.trim().length > 0;
}
