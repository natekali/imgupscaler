import { config, pidConfigured, type Engine, type BrowserQuality } from "./config";
import { downscaleIfNeeded } from "./image-utils";
import { ModalProvider, modalHealthy } from "./providers/modal-client";
import { HuggingFaceProvider } from "./providers/zerogpu-client";
import { BrowserProvider } from "./providers/onnx-realesrgan";
import type { UpscaleProvider, UpscaleResult } from "./providers/types";

export interface UpscaleOptions {
  /** Which engine the user picked. */
  engine: Engine;
  /** Neutral, user-facing status text. */
  onProgress?: (message: string) => void;
  /** Aborts the whole operation (user pressed Cancel). */
  signal?: AbortSignal;
}

/** Thrown when the user cancels; the UI returns to idle, not an error. */
export class CancelledError extends Error {
  constructor() {
    super("Upscale cancelled");
    this.name = "CancelledError";
  }
}

/** Thrown when the NVIDIA PiD engine can't serve (down, timed out, or over quota). */
export class PidUnavailableError extends Error {
  constructor(readonly rateLimited: boolean) {
    super("NVIDIA PiD is unavailable");
    this.name = "PidUnavailableError";
  }
}

/**
 * Upscale with the chosen engine.
 *  - "fast" / "high": run the in-browser Real-ESRGAN model (local, always works).
 *  - "pid": run NVIDIA PiD on a GPU (Modal, then the HF Space). If no PiD backend can
 *    serve, throws PidUnavailableError so the UI can disable the button and message the user.
 */
export async function upscale(file: Blob, options: UpscaleOptions): Promise<UpscaleResult> {
  return options.engine === "pid"
    ? runPid(file, options)
    : runBrowser(file, options.engine, options);
}

async function runBrowser(
  file: Blob,
  quality: BrowserQuality,
  options: UpscaleOptions,
): Promise<UpscaleResult> {
  const model = config.onnxModels[quality];
  const input = await downscaleIfNeeded(file, config.inputCap[quality]);
  const provider = new BrowserProvider(model.label, model.url);

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  options.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const blob = await provider.run(input, {
      signal: controller.signal,
      onProgress: options.onProgress,
    });
    // Show which runtime served the upscale (WebGPU is faster + sharper, WASM is the floor).
    const ep = provider.lastExecutionProvider;
    const label = ep ? `${provider.name} (${ep === "webgpu" ? "WebGPU" : "WASM"})` : provider.name;
    return { blob, engine: label, tier: provider.tier };
  } catch (err) {
    if (options.signal?.aborted) throw new CancelledError();
    throw err;
  } finally {
    options.signal?.removeEventListener("abort", onAbort);
  }
}

async function runPid(file: Blob, options: UpscaleOptions): Promise<UpscaleResult> {
  const providers: UpscaleProvider[] = [new ModalProvider(), new HuggingFaceProvider()].filter(
    (p) => p.isConfigured(),
  );
  if (providers.length === 0) throw new PidUnavailableError(false);

  // PiD center-crops to 512 itself, so a modest cap keeps uploads small without losing detail.
  const input = await downscaleIfNeeded(file, 1024);

  let rateLimited = false;
  for (const provider of providers) {
    if (options.signal?.aborted) throw new CancelledError();

    const controller = new AbortController();
    const onAbort = () => controller.abort();
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => controller.abort(), config.timeouts.pidMs);
    try {
      const blob = await provider.run(input, {
        signal: controller.signal,
        onProgress: options.onProgress,
      });
      return { blob, engine: provider.name, tier: provider.tier };
    } catch (err) {
      if (options.signal?.aborted) throw new CancelledError();
      if ((err as { rateLimited?: boolean }).rateLimited) rateLimited = true;
    } finally {
      options.signal?.removeEventListener("abort", onAbort);
      clearTimeout(timer);
    }
  }
  throw new PidUnavailableError(rateLimited);
}

/**
 * Quick availability probe for the UI: is any PiD backend reachable right now?
 * Used to enable or disable the "NVIDIA PiD" button before the user commits to a slow run.
 */
export async function checkPidAvailable(): Promise<boolean> {
  if (!pidConfigured()) return false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.timeouts.healthMs);
  try {
    if (config.modalBase && (await modalHealthy(controller.signal))) return true;
    // An HF Space can't be cheaply probed without spending quota; treat configured as available
    // and let a real request decide.
    return config.hfSpace.trim().length > 0;
  } finally {
    clearTimeout(timer);
  }
}
