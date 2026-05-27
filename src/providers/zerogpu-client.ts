import { Client, handle_file } from "@gradio/client";
import { config } from "../config";
import type { RunContext, UpscaleProvider } from "./types";

/**
 * Tier 2 — Hugging Face ZeroGPU Space running NVIDIA PiD (free, best-effort accelerator).
 *
 * Called anonymously via @gradio/client, so it draws on HF's shared anonymous GPU quota:
 * it may succeed instantly or be unavailable. Either way the orchestrator fails over to
 * the in-browser engine, so the user never sees a hard error. No token is ever sent from
 * the browser (that would leak it from a public bundle).
 */
export class HuggingFaceProvider implements UpscaleProvider {
  readonly name = "NVIDIA PiD";
  readonly tier = 2;

  isConfigured(): boolean {
    return config.hfSpace.trim().length > 0;
  }

  async run(input: Blob, ctx: RunContext): Promise<Blob> {
    ctx.onProgress?.("Enhancing with NVIDIA PiD…");

    // @gradio/client doesn't take an AbortSignal, so race the call against the signal
    // to keep failover responsive when a tier hangs.
    const work = this.predict(input);
    const aborted = new Promise<never>((_, reject) => {
      if (ctx.signal.aborted) reject(abortError());
      ctx.signal.addEventListener("abort", () => reject(abortError()), { once: true });
    });

    return Promise.race([work, aborted]);
  }

  private async predict(input: Blob): Promise<Blob> {
    const app = await Client.connect(config.hfSpace);
    const result = await app.predict("/upscale", { image: handle_file(input) });

    const out = (result.data as unknown[])[0] as { url?: string } | undefined;
    if (!out?.url) {
      throw new Error("HF Space returned no image");
    }

    const res = await fetch(out.url);
    if (!res.ok) throw new Error(`Failed to fetch HF result (${res.status})`);
    return res.blob();
  }
}

function abortError(): DOMException {
  return new DOMException("Aborted", "AbortError");
}
