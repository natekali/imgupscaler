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
    // PiD is slow (cold GPU boot + diffusion); staged copy keeps the user oriented.
    const stages: Array<[number, string]> = [
      [0, "Sending image to NVIDIA PiD…"],
      [4_000, "Booting GPU (cold start, up to a minute)…"],
      [25_000, "Loading PiD model into VRAM…"],
      [55_000, "Running diffusion (4 steps)…"],
      [85_000, "Finalizing 4K output…"],
    ];
    const start = Date.now();
    const tick = () => {
      const t = Date.now() - start;
      let message = stages[0][1];
      for (const [at, msg] of stages) if (t >= at) message = msg;
      ctx.onProgress?.(message);
    };
    tick();
    const heartbeat = window.setInterval(tick, 900);

    try {
      // Up to two attempts, transparent retry on transient infra blips (502/503/504).
      // 429 (rate-limited) is preserved and surfaced for the UI to disable PiD.
      for (let attempt = 0; attempt < 2; attempt++) {
        const form = new FormData();
        form.append("image", input, "input.png");
        const res = await fetch(`${config.modalBase}/upscale`, {
          method: "POST",
          body: form,
          signal: ctx.signal,
        });
        if (res.ok) {
          const blob = await res.blob();
          if (!blob.type.startsWith("image/")) {
            throw new Error("Modal PiD returned a non-image response");
          }
          return blob;
        }
        const transient = res.status === 502 || res.status === 503 || res.status === 504;
        if (transient && attempt === 0) {
          await new Promise((r) => setTimeout(r, 2_000));
          continue;
        }
        const err = new Error(`Modal PiD responded ${res.status}`) as Error & {
          rateLimited?: boolean;
        };
        // 429 = over quota / rate limit. Surface so the UI can disable the button.
        err.rateLimited = res.status === 429;
        throw err;
      }
      throw new Error("Modal PiD: unreachable");
    } finally {
      window.clearInterval(heartbeat);
    }
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
