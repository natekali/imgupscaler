import type * as Ort from "onnxruntime-web";
import type { RunContext, UpscaleProvider } from "./types";

// onnxruntime-web is loaded from a CDN on demand (not bundled): it's only needed when the
// in-browser tier actually runs, and bundling it would ship a ~26 MB wasm we don't use.
const ORT_VERSION = "1.20.1";
const ORT_ESM = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const ORT_WASM_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const SCALE = 4; // Real-ESRGAN x4
const TILE = 128; // core tile edge (px) — bounds GPU/WASM memory
const PAD = 16; // context padding read around each tile, discarded after inference (kills seams)

type OrtModule = typeof import("onnxruntime-web");
let ortPromise: Promise<OrtModule> | null = null;

async function loadOrt(): Promise<OrtModule> {
  if (!ortPromise) {
    ortPromise = (async () => {
      const ort = (await import(/* @vite-ignore */ ORT_ESM)) as OrtModule;
      ort.env.wasm.wasmPaths = ORT_WASM_DIR;
      return ort;
    })();
  }
  return ortPromise;
}

// Sessions are cached per model URL so switching quality back and forth doesn't reload.
const sessions = new Map<string, Promise<Ort.InferenceSession>>();

function getSession(ort: OrtModule, url: string): Promise<Ort.InferenceSession> {
  const cached = sessions.get(url);
  if (cached) return cached;
  const attempt = (async () => {
    try {
      return await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
    } catch {
      return await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    }
  })();
  // Don't cache a rejected load — a transient network failure would otherwise poison every
  // later attempt until a full reload. Only memoize the session once it resolves.
  attempt.catch(() => {
    if (sessions.get(url) === attempt) sessions.delete(url);
  });
  sessions.set(url, attempt);
  return attempt;
}

/**
 * Tier 3 — Real-ESRGAN x4 running entirely in the browser via onnxruntime-web.
 *
 * This engine can never be "busy" or rate-limited: it is the unconditional floor that
 * guarantees every upload returns a result, even fully offline once the model is cached.
 * Quality is below PiD, but it always works. Runtime + model are lazy-loaded on first use.
 */
export class BrowserProvider implements UpscaleProvider {
  readonly tier = 3;

  constructor(
    readonly name: string,
    private readonly modelUrl: string,
  ) {}

  isConfigured(): boolean {
    return this.modelUrl.trim().length > 0;
  }

  async run(input: Blob, ctx: RunContext): Promise<Blob> {
    ctx.onProgress?.("Enhancing locally in your browser…");
    const ort = await loadOrt();
    const session = await getSession(ort, this.modelUrl);
    throwIfAborted(ctx.signal);

    const src = await blobToImageData(input);
    const { width: w, height: h } = src;

    const outCanvas = new OffscreenCanvas(w * SCALE, h * SCALE);
    const outCtx = outCanvas.getContext("2d");
    if (!outCtx) throw new Error("Canvas 2D context unavailable");

    const cols = Math.ceil(w / TILE);
    const rows = Math.ceil(h / TILE);
    const total = cols * rows;
    let done = 0;

    for (let ty = 0; ty < rows; ty++) {
      for (let tx = 0; tx < cols; tx++) {
        throwIfAborted(ctx.signal);

        // Core region for this tile (clamped to image bounds).
        const cx = tx * TILE;
        const cy = ty * TILE;
        const coreW = Math.min(TILE, w - cx);
        const coreH = Math.min(TILE, h - cy);

        // Read region = core expanded by PAD, clamped. Padding gives the model context so
        // tile borders are seam-free; we crop the padding back off after inference.
        const rx = Math.max(0, cx - PAD);
        const ry = Math.max(0, cy - PAD);
        const rw = Math.min(w, cx + coreW + PAD) - rx;
        const rh = Math.min(h, cy + coreH + PAD) - ry;
        const padLeft = cx - rx;
        const padTop = cy - ry;

        const tensor = new ort.Tensor("float32", regionToCHW(src, rx, ry, rw, rh), [1, 3, rh, rw]);
        const feeds: Record<string, Ort.Tensor> = { [session.inputNames[0]]: tensor };
        const result = await session.run(feeds);
        const out = result[session.outputNames[0]];

        // Crop the padded border out of the SR output and paste the valid core.
        const [, , outH, outW] = out.dims as number[];
        const tileImage = chwToImageData(
          out.data as Float32Array,
          outW,
          outH,
          padLeft * SCALE,
          padTop * SCALE,
          coreW * SCALE,
          coreH * SCALE,
        );
        outCtx.putImageData(tileImage, cx * SCALE, cy * SCALE);

        done++;
        ctx.onProgress?.(`Enhancing locally… ${Math.round((done / total) * 100)}%`);
      }
    }

    return outCanvas.convertToBlob({ type: "image/png" });
  }
}

/* ---------- pixel <-> tensor helpers (ort-agnostic) ---------- */

async function blobToImageData(blob: Blob): Promise<ImageData> {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/** Extract an RGB region as a normalized CHW float32 buffer (0..1). */
function regionToCHW(img: ImageData, rx: number, ry: number, rw: number, rh: number): Float32Array {
  const data = new Float32Array(3 * rw * rh);
  const plane = rw * rh;
  const stride = img.width * 4;
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const srcIdx = (ry + y) * stride + (rx + x) * 4;
      const dstIdx = y * rw + x;
      data[dstIdx] = img.data[srcIdx] / 255; // R
      data[plane + dstIdx] = img.data[srcIdx + 1] / 255; // G
      data[2 * plane + dstIdx] = img.data[srcIdx + 2] / 255; // B
    }
  }
  return data;
}

/** Convert a CHW float32 SR buffer (0..1) into an ImageData, cropping to a sub-rect. */
function chwToImageData(
  src: Float32Array,
  W: number,
  H: number,
  offsetX: number,
  offsetY: number,
  cropW: number,
  cropH: number,
): ImageData {
  const plane = H * W;
  const out = new ImageData(cropW, cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const sIdx = (offsetY + y) * W + (offsetX + x);
      const dIdx = (y * cropW + x) * 4;
      out.data[dIdx] = clamp8(src[sIdx]);
      out.data[dIdx + 1] = clamp8(src[plane + sIdx]);
      out.data[dIdx + 2] = clamp8(src[2 * plane + sIdx]);
      out.data[dIdx + 3] = 255;
    }
  }
  return out;
}

function clamp8(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException("Aborted", "AbortError");
}
