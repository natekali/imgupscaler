import "./styles/main.css";
import "img-comparison-slider";
import { upscale } from "./upscale-orchestrator";
import { getDimensions } from "./image-utils";

/* ------------------------------------------------------------------ *
 * Resolve — UI state machine.
 * States: idle → processing → (result | error). The in-browser engine
 * means "error" is rare; it's there only for truly broken inputs.
 * ------------------------------------------------------------------ */

const MAX_FILE_BYTES = 25 * 1024 * 1024; // 25 MB upload cap
const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

type State = "idle" | "processing" | "result" | "error";

// Every object URL we mint is tracked here and revoked on reset / unload, so an image
// never lingers in memory longer than the user needs it.
const liveUrls = new Set<string>();
function objectUrl(blob: Blob): string {
  const url = URL.createObjectURL(blob);
  liveUrls.add(url);
  return url;
}
function revokeAll(): void {
  for (const url of liveUrls) URL.revokeObjectURL(url);
  liveUrls.clear();
}

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`Missing element: ${sel}`);
  return el;
};

const stage = $<HTMLElement>(".stage");
const fileInput = $<HTMLInputElement>("#file-input");
const dropzone = $<HTMLLabelElement>("#dropzone");
const procImg = $<HTMLImageElement>("#proc-img");
const procStatus = $<HTMLElement>("#proc-status");
const procBarFill = $<HTMLElement>("#proc-bar-fill");
const compareWrap = $<HTMLElement>("#compare-wrap");
const resultMeta = $<HTMLElement>("#result-meta");
const downloadBtn = $<HTMLButtonElement>("#download-btn");
const resetBtn = $<HTMLButtonElement>("#reset-btn");
const errMsg = $<HTMLElement>("#err-msg");
const errReset = $<HTMLButtonElement>("#err-reset");

let resultBlob: Blob | null = null;

function setState(state: State): void {
  stage.dataset.state = state;
}

/* ---------------- file intake ---------------- */

function validate(file: File): string | null {
  if (!ACCEPTED.includes(file.type)) return "Please use a PNG, JPG, or WebP image.";
  if (file.size > MAX_FILE_BYTES) return "That image is over 25 MB — please use a smaller file.";
  return null;
}

async function handleFile(file: File): Promise<void> {
  const problem = validate(file);
  if (problem) return showError(problem);

  setState("processing");
  procImg.src = objectUrl(file);
  setProgress("Initializing engine…");

  try {
    const { blob, engine, tier } = await upscale(file, { onProgress: setProgress });
    resultBlob = blob;
    await showResult(file, blob, engine, tier);
  } catch (err) {
    showError(err instanceof Error ? err.message : "Something went wrong. Try another image.");
  }
}

/* ---------------- progress ---------------- */

function setProgress(message: string): void {
  const pct = message.match(/(\d{1,3})%/);
  if (pct) {
    procBarFill.dataset.indeterminate = "false";
    procBarFill.style.width = `${Math.min(100, Number(pct[1]))}%`;
  } else {
    procBarFill.dataset.indeterminate = "true";
  }
  procStatus.textContent = message;
}

/* ---------------- result ---------------- */

async function showResult(original: Blob, result: Blob, engine: string, tier: number): Promise<void> {
  const dims = await getDimensions(result);

  compareWrap.style.position = "relative";
  compareWrap.replaceChildren();

  const slider = document.createElement("img-comparison-slider");
  const before = document.createElement("img");
  before.slot = "first";
  before.src = objectUrl(original);
  before.alt = "Original image";
  const after = document.createElement("img");
  after.slot = "second";
  after.src = objectUrl(result);
  after.alt = "Upscaled image";
  slider.append(before, after);

  // Credit the engine that actually ran — don't claim PiD when the in-browser floor produced it.
  const beforeTag = tag("Before", "cmp-tag--before");
  const afterTag = tag(`After · ${engine.replace(" (in-browser)", "")}`, "cmp-tag--after");
  compareWrap.append(slider, beforeTag, afterTag);

  resultMeta.replaceChildren();
  resultMeta.append(
    metaItem("Engine", engine),
    metaItem("Output", `${dims.width} × ${dims.height}`),
    metaItem("Scale", "4×", true),
  );
  // tier is reflected through the engine label; kept for future telemetry-free debugging.
  void tier;

  setState("result");
}

function tag(text: string, cls: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = `cmp-tag ${cls}`;
  span.textContent = text;
  return span;
}

function metaItem(label: string, value: string, signal = false): HTMLElement {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (signal) dd.className = "signal";
  wrap.append(dt, dd);
  return wrap;
}

/* ---------------- download / reset / error ---------------- */

function download(): void {
  if (!resultBlob) return;
  const url = URL.createObjectURL(resultBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `resolve-4k-${Date.now()}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Release immediately after the download is triggered — the file is the user's now.
  URL.revokeObjectURL(url);
}

function reset(): void {
  revokeAll();
  resultBlob = null;
  fileInput.value = "";
  procImg.removeAttribute("src");
  compareWrap.innerHTML = "";
  setProgress("Initializing engine…");
  setState("idle");
}

function showError(message: string): void {
  errMsg.textContent = message;
  setState("error");
}

/* ---------------- events ---------------- */

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void handleFile(file);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  }),
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  }),
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) void handleFile(file);
});

// Allow dropping anywhere on the page while idle, without the browser navigating away.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

downloadBtn.addEventListener("click", download);
resetBtn.addEventListener("click", reset);
errReset.addEventListener("click", reset);
window.addEventListener("beforeunload", revokeAll);
