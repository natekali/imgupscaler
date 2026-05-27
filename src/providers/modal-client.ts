import { config } from "../config";
import type { RunContext, UpscaleProvider } from "./types";

/**
 * NVIDIA PiD running on Modal (serverless GPU). Posts the image as multipart/form-data to
 * the CORS-enabled endpoint and gets back the upscaled PNG. No credentials travel from the
 * browser: the Modal function authenticates server-side. Stateless: temp files are deleted
 * per request, nothing is stored.
 */
export class ModalProvider implements UpscaleProvider {
  readonly name = "NVIDIA PiD";
  readonly tier = 1;

  isConfigured(): boolean {
    return config.modalBase.trim().length > 0;
  }

  async run(input: Blob, ctx: RunContext): Promise<Blob> {
    ctx.onProgress?.("Enhancing with NVIDIA PiD… (GPU warming up, up to a minute)");

    const form = new FormData();
    form.append("image", input, "input.png");

    const res = await fetch(`${config.modalBase}/upscale`, {
      method: "POST",
      body: form,
      signal: ctx.signal,
    });

    if (!res.ok) {
      const err = new Error(`Modal PiD responded ${res.status}`) as Error & { rateLimited?: boolean };
      // 429 = over quota, 503 = scaling/at capacity: both mean "come back later".
      err.rateLimited = res.status === 429 || res.status === 503;
      throw err;
    }

    const blob = await res.blob();
    if (!blob.type.startsWith("image/")) {
      throw new Error("Modal PiD returned a non-image response");
    }
    return blob;
  }
}

/** Cheap liveness probe used to enable/disable the PiD button in the UI. */
export async function modalHealthy(signal: AbortSignal): Promise<boolean> {
  if (!config.modalBase.trim()) return false;
  try {
    const res = await fetch(`${config.modalBase}/health`, { signal });
    return res.ok;
  } catch {
    return false;
  }
}
