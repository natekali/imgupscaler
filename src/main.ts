import "./styles/main.css";
import "img-comparison-slider";
import { upscale, CancelledError, PidUnavailableError, checkPidAvailable } from "./upscale-orchestrator";
import { getDimensions } from "./image-utils";
import { pidConfigured, type Engine } from "./config";

/* ------------------------------------------------------------------ *
 * Upscaler AI, UI state machine.
 * States: idle → processing → (result | error). The in-browser engine
 * means "error" is rare; it's there only for truly broken inputs.
 * ------------------------------------------------------------------ */

const MAX_FILE_BYTES = 25 * 1024 * 1024; // 25 MB upload cap
const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

type State = "idle" | "processing" | "result" | "error";
type Format = "png" | "jpeg";

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
const cancelBtn = $<HTMLButtonElement>("#cancel-btn");
const compareWrap = $<HTMLElement>("#compare-wrap");
const resultMeta = $<HTMLElement>("#result-meta");
const downloadBtn = $<HTMLButtonElement>("#download-btn");
const copyBtn = $<HTMLButtonElement>("#copy-btn");
const resetBtn = $<HTMLButtonElement>("#reset-btn");
const errMsg = $<HTMLElement>("#err-msg");
const errReset = $<HTMLButtonElement>("#err-reset");
const pidBtn = document.querySelector<HTMLButtonElement>('.q-opt[data-engine="pid"]');
const pidDesc = document.querySelector<HTMLElement>("#pid-desc");
const engineNotice = $<HTMLElement>("#engine-notice");

let engine: Engine = "fast";
let pidDisabled = false; // set when PiD is confirmed unavailable / over quota
let format: Format = "png";
let pngBlob: Blob | null = null; // raw engine output (PNG)
let downloadBlob: Blob | null = null; // possibly re-encoded for the chosen format
let sizeEl: HTMLElement | null = null; // meta cell we live-update on format change
let activeRun: AbortController | null = null;

function setState(state: State): void {
  stage.dataset.state = state;
}

/* ---------------- file intake ---------------- */

function validate(file: File): string | null {
  if (!ACCEPTED.includes(file.type)) return "Please use a PNG, JPG, or WebP image.";
  if (file.size > MAX_FILE_BYTES) return "That image is over 25 MB. Please use a smaller file.";
  return null;
}

async function handleFile(file: File): Promise<void> {
  const problem = validate(file);
  if (problem) return showError(problem);

  setState("processing");
  procImg.src = objectUrl(file);
  setProgress(engine === "pid" ? "Sending to NVIDIA PiD…" : "Initializing engine…");
  clearNotice();

  activeRun = new AbortController();
  try {
    const result = await upscale(file, { engine, signal: activeRun.signal, onProgress: setProgress });
    await showResult(file, result.blob, result.engine);
  } catch (err) {
    if (err instanceof CancelledError) {
      reset(); // user bailed, quietly return to idle
      return;
    }
    if (err instanceof PidUnavailableError) {
      // PiD is down or over quota: disable the button, tell the user, and don't strand them.
      markPidUnavailable(err.rateLimited);
      setProgress("NVIDIA PiD is busy, finishing locally…");
      try {
        const result = await upscale(file, { engine: "fast", signal: activeRun.signal, onProgress: setProgress });
        await showResult(file, result.blob, result.engine);
      } catch (err2) {
        if (err2 instanceof CancelledError) reset();
        else showError("Something went wrong. Try another image.");
      }
      return;
    }
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

async function showResult(original: Blob, result: Blob, engine: string): Promise<void> {
  pngBlob = result;
  const [inputDims, dims] = await Promise.all([getDimensions(original), getDimensions(result)]);

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

  // Credit the engine that actually ran, don't claim PiD when the in-browser floor produced it.
  const beforeTag = tag("Before", "cmp-tag--before");
  const afterTag = tag(`After · ${engine}`, "cmp-tag--after");
  compareWrap.append(slider, beforeTag, afterTag);

  const size = metaItem("Size", formatBytes(result.size));
  sizeEl = size.querySelector("dd");
  resultMeta.replaceChildren(
    metaItem("Engine", engine),
    metaItem("Input", `${inputDims.width} × ${inputDims.height}`),
    metaItem("Output", `${dims.width} × ${dims.height}`, true),
    size,
  );

  format = "png";
  syncFormatButtons();
  downloadBlob = result;
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

function formatBytes(n: number): string {
  return n >= 1024 * 1024
    ? `${(n / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(n / 1024))} KB`;
}

/** Re-encode the PNG output to the chosen format and refresh the displayed size. */
async function prepareDownload(): Promise<void> {
  if (!pngBlob) return;
  if (format === "png") {
    downloadBlob = pngBlob;
  } else {
    const bitmap = await createImageBitmap(pngBlob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
    bitmap.close();
    downloadBlob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 });
  }
  if (sizeEl) sizeEl.textContent = formatBytes(downloadBlob.size);
}

/* ---------------- download / reset / error ---------------- */

function download(): void {
  if (!downloadBlob) return;
  const url = URL.createObjectURL(downloadBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `upscaler-ai-4k-${Date.now()}.${format === "jpeg" ? "jpg" : "png"}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer the revoke: revoking synchronously can abort the save in Firefox/Safari before
  // the download stream starts. A short delay lets the browser take ownership first.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function copyToClipboard(): Promise<void> {
  if (!pngBlob) return;
  const supported = "clipboard" in navigator && typeof ClipboardItem !== "undefined";
  if (!supported) return flashButton(copyBtn, "Not supported", true);
  try {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": pngBlob })]);
    flashButton(copyBtn, "Copied!");
  } catch {
    flashButton(copyBtn, "Couldn't copy", true);
  }
}

/** Briefly swap the button label to signal success or failure, then restore. */
function flashButton(btn: HTMLButtonElement, message: string, isError = false): void {
  const original = btn.textContent;
  btn.textContent = message;
  btn.classList.toggle("flash-error", isError);
  window.setTimeout(() => {
    btn.textContent = original;
    btn.classList.remove("flash-error");
  }, 1500);
}

function reset(): void {
  activeRun?.abort();
  activeRun = null;
  revokeAll();
  pngBlob = null;
  downloadBlob = null;
  sizeEl = null;
  fileInput.value = "";
  procImg.removeAttribute("src");
  compareWrap.replaceChildren();
  setProgress("Initializing engine…");
  setState("idle");
}

function showError(message: string): void {
  errMsg.textContent = message;
  setState("error");
}

/* ---------------- toggles ---------------- */

function syncFormatButtons(): void {
  document.querySelectorAll<HTMLButtonElement>(".fmt-opt").forEach((b) => {
    const active = b.dataset.fmt === format;
    b.classList.toggle("is-active", active);
    b.setAttribute("aria-checked", String(active));
  });
}

function setEngineActive(target: Engine): void {
  engine = target;
  document.querySelectorAll<HTMLButtonElement>(".q-opt").forEach((b) => {
    const active = b.dataset.engine === target;
    b.classList.toggle("is-active", active);
    b.setAttribute("aria-checked", String(active));
  });
}

function showNotice(message: string): void {
  engineNotice.textContent = message;
  engineNotice.hidden = false;
}
function clearNotice(): void {
  engineNotice.hidden = true;
}

/** Disable the PiD button and tell the user to come back later. */
function markPidUnavailable(rateLimited: boolean): void {
  pidDisabled = true;
  pidBtn?.classList.add("is-disabled");
  pidBtn?.setAttribute("aria-disabled", "true");
  if (pidDesc) pidDesc.textContent = "At capacity, come back later";
  showNotice(
    rateLimited
      ? "NVIDIA PiD has hit its rate limit. Please come back later once it resets."
      : "NVIDIA PiD is unavailable right now. Please come back later.",
  );
  if (engine === "pid") setEngineActive("fast");
}

// Bumped on every engine click so a slow PiD health check can't override a later choice.
let selectSeq = 0;
let pidChecking = false;

async function selectPid(seq: number): Promise<void> {
  if (pidDisabled || pidChecking) return;
  if (!pidConfigured()) return markPidUnavailable(false);
  pidChecking = true;
  if (pidDesc) pidDesc.textContent = "Checking availability…";
  const available = await checkPidAvailable();
  pidChecking = false;
  if (seq !== selectSeq) {
    // The user picked a different engine while we were checking; respect that.
    if (pidDesc) pidDesc.textContent = "Best quality · GPU diffusion";
    return;
  }
  if (available) {
    if (pidDesc) pidDesc.textContent = "Best quality · GPU diffusion";
    clearNotice();
    setEngineActive("pid");
  } else {
    markPidUnavailable(false);
  }
}

document.querySelectorAll<HTMLButtonElement>(".q-opt").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.engine as Engine;
    const seq = ++selectSeq;
    if (target === "pid") void selectPid(seq);
    else {
      clearNotice();
      setEngineActive(target);
    }
  });
});

// If no PiD backend is configured at all, present the option as unavailable up front.
if (!pidConfigured()) {
  pidDisabled = true;
  pidBtn?.classList.add("is-disabled");
  pidBtn?.setAttribute("aria-disabled", "true");
  if (pidDesc) pidDesc.textContent = "Not deployed";
}

document.querySelectorAll<HTMLButtonElement>(".fmt-opt").forEach((btn) => {
  btn.addEventListener("click", () => {
    format = btn.dataset.fmt as Format;
    syncFormatButtons();
    void prepareDownload();
  });
});

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

// Paste an image straight from the clipboard (Ctrl/Cmd+V).
window.addEventListener("paste", (e) => {
  const item = Array.from(e.clipboardData?.items ?? []).find(
    (i) => i.kind === "file" && i.type.startsWith("image/"),
  );
  const file = item?.getAsFile();
  if (file) {
    if (stage.dataset.state !== "idle") reset();
    void handleFile(file);
  }
});

cancelBtn.addEventListener("click", reset);
downloadBtn.addEventListener("click", download);
copyBtn.addEventListener("click", () => void copyToClipboard());
resetBtn.addEventListener("click", reset);
errReset.addEventListener("click", reset);
window.addEventListener("beforeunload", revokeAll);
