import { config } from "./config";
import { downscaleIfNeeded } from "./image-utils";
import { ModalProvider } from "./providers/modal-client";
import { HuggingFaceProvider } from "./providers/zerogpu-client";
import { BrowserProvider } from "./providers/onnx-realesrgan";
import type { UpscaleProvider, UpscaleResult } from "./providers/types";

export interface OrchestratorEvents {
  /** Neutral, user-facing status text. Never says "busy" / "not ready". */
  onProgress?: (message: string) => void;
}

/**
 * Run the failover chain: Tier 1 (Modal/PiD) → Tier 2 (HF Space/PiD) → Tier 3 (in-browser).
 *
 * Each remote tier gets a wall-clock timeout; on timeout or error we transparently fall
 * through to the next engine. The in-browser tier is always configured and never rejects
 * for capacity reasons, so a result is always produced — the user never sees a hard
 * "engine unavailable" failure.
 */
export async function upscale(
  file: Blob,
  events: OrchestratorEvents = {},
): Promise<UpscaleResult> {
  const providers = [new ModalProvider(), new HuggingFaceProvider(), new BrowserProvider()]
    .filter((p) => p.isConfigured())
    .sort((a, b) => a.tier - b.tier);

  // Cap the longest edge before any engine runs. This saves GPU seconds on the remote tiers
  // and — critically — bounds the in-browser tier's 4× output canvas so a large upload can't
  // OOM the tab (the browser tier is the active path until a GPU backend is configured).
  const input = await downscaleIfNeeded(file, config.maxRemoteInputEdge);

  let lastError: unknown;
  for (const provider of providers) {
    const timeoutMs = timeoutFor(provider);
    const controller = new AbortController();
    const timer =
      timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : undefined;

    try {
      const blob = await provider.run(input, {
        signal: controller.signal,
        onProgress: events.onProgress,
      });
      return { blob, engine: provider.name, tier: provider.tier };
    } catch (err) {
      lastError = err;
      // Try the next engine without alarming the user.
      events.onProgress?.("Switching to the next engine…");
    } finally {
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
