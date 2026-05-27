import { config } from "../config";
import type { RunContext, UpscaleProvider } from "./types";

/**
 * Tier 1, Modal serverless GPU running NVIDIA PiD (reliable primary).
 *
 * Posts the image as multipart/form-data to the CORS-enabled Modal endpoint and gets
 * back the upscaled PNG. No credentials travel from the browser: the Modal function
 * authenticates server-side. The endpoint is stateless, it deletes its temp files per
 * request, so nothing is stored.
 */
export class ModalProvider implements UpscaleProvider {
  readonly name = "NVIDIA PiD";
  readonly tier = 1;

  isConfigured(): boolean {
    return config.modalEndpoint.trim().length > 0;
  }

  async run(input: Blob, ctx: RunContext): Promise<Blob> {
    ctx.onProgress?.("Enhancing with NVIDIA PiD…");

    const form = new FormData();
    form.append("image", input, "input.png");

    const res = await fetch(config.modalEndpoint, {
      method: "POST",
      body: form,
      signal: ctx.signal,
    });

    if (!res.ok) {
      throw new Error(`Modal backend responded ${res.status}`);
    }

    const blob = await res.blob();
    if (!blob.type.startsWith("image/")) {
      throw new Error("Modal backend returned a non-image response");
    }
    return blob;
  }
}
