import type * as Ort from "onnxruntime-web";
import type { RunContext, UpscaleProvider } from "./types";

// onnxruntime-web is loaded from a CDN on demand (not bundled): it's only needed when the
// in-browser tier actually runs, and bundling it would ship a ~26 MB wasm we don't use.
const ORT_VERSION = "1.20.1";
const ORT_ESM = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const ORT_WASM_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const SCALE = 4; // Real-ESRGAN x4
// Tile + padding sized per execution provider. WebGPU handles bigger tiles fast,
// WASM stays at the smaller original tile to keep memory and per-tile time reasonable.
const TILE_WEBGPU = 256;
const PAD_WEBGPU = 24;
const TILE_WASM = 128;
const PAD_WASM = 16;

type OrtModule = typeof import("onnxruntime-web");
type ExecutionProvider = "webgpu" | "wasm";

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

// One-shot WebGPU support detection, cached. Avoids re-throwing through onnxruntime-web on
// every session create and lets the UI know which runtime to expect.
let webgpuPromise: Promise<boolean> | null = null;
function webgpuSupported(): Promise<boolean> {
  if (webgpuPromise === null) {
    webgpuPromise = (async () => {
      try {
        const nav = navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } };
        if (!nav.gpu) return false;
        return !!(await nav.gpu.requestAdapter());
      } catch {
        return false;
      }
    })();
  }
  return webgpuPromise;
}

interface CachedSession {
  session: Ort.InferenceSession;
  ep: ExecutionProvider;
}

// Sessions are cached per model URL so switching quality back and forth doesn't reload.
const sessions = new Map<string, Promise<CachedSession>>();

function getSession(ort: OrtModule, url: string): Promise<CachedSession> {
  const cached = sessions.get(url);
  if (cached) return cached;
  const attempt: Promise<CachedSession> = (async () => {
    if (await webgpuSupported()) {
      try {
        const session = await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
        return { session, ep: "webgpu" as const };
      } catch {
        // Fall through to WASM.
      }
    }
    const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    return { session, ep: "wasm" as const };
  })();
  // Don't cache a rejected load: a transient network failure would otherwise poison every
  // later attempt until a full reload. Only memoize once it resolves.
  attempt.catch(() => {
    if (sessions.get(url) === attempt) sessions.delete(url);
  });
  sessions.set(url, attempt);
  return attempt;
}

/**
 * Tier 3, Real-ESRGAN x4 running entirely in the browser via onnxruntime-web.
 *
 * Cannot be "busy" or rate-limited: this is the unconditional floor that guarantees every
 * upload returns a result, even fully offline once the model is cached. Runtime + model are
 * lazy-loaded on first use; sessions per quality are cached for instant repeat runs.
 */
export class BrowserProvider implements UpscaleProvider {
  readonly tier = 3;
  /** Set after a successful run so the UI can surface "WebGPU" vs "WASM" in the result meta. */
  lastExecutionProvider: ExecutionProvider | null = null;

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
    const cached = await getSession(ort, this.modelUrl);
    const { session, ep } = cached;
    this.lastExecutionProvider = ep;
    throwIfAborted(ctx.signal);

    const src = await blobToImageData(input);
    const { width: w, height: h } = src;
    const hasAlpha = detectAlpha(src);

    const outCanvas = new OffscreenCanvas(w * SCALE, h * SCALE);
    const outCtx = outCanvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    if (!outCtx) throw new Error("Canvas 2D context unavailable");

    const tile = ep === "webgpu" ? TILE_WEBGPU : TILE_WASM;
    const pad = ep === "webgpu" ? PAD_WEBGPU : PAD_WASM;
    const cols = Math.ceil(w / tile);
    const rows = Math.ceil(h / tile);
    const total = cols * rows;
    let done = 0;

    for (let ty = 0; ty < rows; ty++) {
      for (let tx = 0; tx < cols; tx++) {
        throwIfAborted(ctx.signal);

        // Core region for this tile (clamped to image bounds).
        const cx = tx * tile;
        const cy = ty * tile;
        const coreW = Math.min(tile, w - cx);
        const coreH = Math.min(tile, h - cy);

        // Read region = core expanded by pad, clamped. Padding gives the model context so
        // tile borders are seam-free; we crop the padding back off after inference.
        const rx = Math.max(0, cx - pad);
        const ry = Math.max(0, cy - pad);
        const rw = Math.min(w, cx + coreW + pad) - rx;
        const rh = Math.min(h, cy + coreH + pad) - ry;
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

    // The model upscales RGB only; if the source had transparency, upscale its alpha channel
    // separately (bilinear) and re-apply it so transparent PNGs don't composite onto black.
    if (hasAlpha) {
      await applyUpscaledAlpha(input, outCtx, w * SCALE, h * SCALE);
    }

    return outCanvas.convertToBlob({ type: "image/png" });
  }
}

/** True if any pixel is not fully opaque. */
function detectAlpha(img: ImageData): boolean {
  for (let i = 3; i < img.data.length; i += 4) {
    if (img.data[i] < 255) return true;
  }
  return false;
}

/** Bilinearly upscale the source's alpha channel and write it into the SR output. */
async function applyUpscaledAlpha(
  source: Blob,
  outCtx: OffscreenCanvasRenderingContext2D,
  outW: number,
  outH: number,
): Promise<void> {
  // Straight alpha + no color-space munging keeps the values the model saw.
  const bitmap = await createImageBitmap(source, {
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  const alphaCanvas = new OffscreenCanvas(outW, outH);
  const ac = alphaCanvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
  if (!ac) return;
  ac.imageSmoothingEnabled = true;
  ac.imageSmoothingQuality = "high";
  ac.drawImage(bitmap, 0, 0, outW, outH);
  bitmap.close();

  const scaled = ac.getImageData(0, 0, outW, outH);
  const out = outCtx.getImageData(0, 0, outW, outH);
  for (let i = 3; i < out.data.length; i += 4) {
    out.data[i] = scaled.data[i];
  }
  outCtx.putImageData(out, 0, 0);
}

/* ---------- pixel <-> tensor helpers (ort-agnostic) ---------- */

async function blobToImageData(blob: Blob): Promise<ImageData> {
  // Straight alpha (not premultiplied) and no implicit color conversion so the tensor
  // values match the source pixel values the model was trained on.
  const bitmap = await createImageBitmap(blob, {
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
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
