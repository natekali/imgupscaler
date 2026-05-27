import { config, type Quality } from "./config";
import { downscaleIfNeeded } from "./image-utils";
import { ModalProvider } from "./providers/modal-client";
import { HuggingFaceProvider } from "./providers/zerogpu-client";
import { BrowserProvider } from "./providers/onnx-realesrgan";
import type { UpscaleProvider, UpscaleResult } from "./providers/types";

export interface OrchestratorOptions {
  /** Neutral, user-facing status text. Never says "busy" / "not ready". */
  onProgress?: (message: string) => void;
  /** Which in-browser model to use when the browser tier runs. */
  quality?: Quality;
  /** Aborts the whole operation (user pressed Cancel). */
  signal?: AbortSignal;
}

/** Thrown when the user cancels; the UI treats this as "return to idle", not an error. */
export class CancelledError extends Error {
  constructor() {
    super("Upscale cancelled");
    this.name = "CancelledError";
  }
}

/**
 * Run the failover chain: Tier 1 (Modal/PiD) → Tier 2 (HF Space/PiD) → Tier 3 (in-browser).
 *
 * Each remote tier gets a wall-clock timeout; on timeout or error we transparently fall
 * through to the next engine. The in-browser tier is always configured and never rejects
 * for capacity reasons, so a result is always produced, the user never sees a hard
 * "engine unavailable" failure (unless they cancel).
 */
export async function upscale(
  file: Blob,
  options: OrchestratorOptions = {},
): Promise<UpscaleResult> {
  const quality = options.quality ?? config.defaultQuality;
  const browser = config.onnxModels[quality];

  const providers: UpscaleProvider[] = [
    new ModalProvider(),
    new HuggingFaceProvider(),
    new BrowserProvider(browser.label, browser.url),
  ]
    .filter((p) => p.isConfigured())
    .sort((a, b) => a.tier - b.tier);

  // Cap the longest edge before any engine runs. Saves GPU seconds on remote tiers and -
  // critically, bounds the in-browser tier's 4× output canvas so a large upload can't OOM
  // the tab (the browser tier is the active path until a GPU backend is configured).
  const input = await downscaleIfNeeded(file, config.inputCap[quality]);

  let lastError: unknown;
  for (const provider of providers) {
    if (options.signal?.aborted) throw new CancelledError();

    const timeoutMs = timeoutFor(provider);
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const timer =
      timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : undefined;

    try {
      const blob = await provider.run(input, {
        signal: controller.signal,
        onProgress: options.onProgress,
      });
      return { blob, engine: provider.name, tier: provider.tier };
    } catch (err) {
      // A user cancel aborts the whole chain; a timeout/error just moves to the next engine.
      if (options.signal?.aborted) throw new CancelledError();
      lastError = err;
      options.onProgress?.("Switching to the next engine…");
    } finally {
      options.signal?.removeEventListener("abort", onAbort);
      if (timer) clearTimeout(timer);
    }
  }

  throw new Error(
    `All upscaling engines failed${lastError ? `: ${String(lastError)}` : ""}`,
  );
}

function timeoutFor(provider: UpscaleProvider): number {
  switch (provider.tier) {
    case 1:
      return config.timeouts.modalMs;
    case 2:
      return config.timeouts.hfSpaceMs;
    default:
      return 0; // in-browser: no timeout, it always finishes
  }
}
