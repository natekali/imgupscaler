<div align="center">

# Upscaler AI

**Free AI Image Upscaler, powered by NVIDIA PiD**

*Drop a low-resolution image, get a crisp, 4K-grade result in seconds. Right in your browser, nothing to install, nothing stored.*

![License](https://img.shields.io/badge/License-Apache%202.0-d4ff3d.svg)
![Built with TypeScript](https://img.shields.io/badge/built%20with-TypeScript-3178c6.svg)
![Bundler](https://img.shields.io/badge/bundler-Vite-646cff.svg)
![Powered by NVIDIA PiD](https://img.shields.io/badge/powered%20by-NVIDIA%20PiD-76b900.svg)
![GPU](https://img.shields.io/badge/GPU-not%20required-d4ff3d.svg)

[Live Demo](https://natekali.github.io/imgupscaler/) | [Features](#-features) | [How It Works](#-how-it-works) | [Quick Start](#-quick-start) | [Privacy](#-privacy) | [Deploy](#%EF%B8%8F-deploy)

<img src="docs/preview.png" alt="Upscaler AI" width="820" />

</div>

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [What is Upscaler AI](#what-is-upscaler-ai)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Privacy](#-privacy)
- [Quick Start](#-quick-start)
- [Deploy](#%EF%B8%8F-deploy)
- [Optional: the PiD GPU tiers](#optional-the-pid-gpu-tiers)
- [Tech Stack](#-tech-stack)
- [Credits](#-credits)
- [License](#-license)

</details>

---

## What is Upscaler AI

Upscaler AI is a dead-simple web app for image super-resolution: you drop in a small or blurry
image and it returns a sharp, high-resolution version at up to 4x. It is headlined by NVIDIA's
brand-new **[PiD](https://research.nvidia.com/labs/sil/projects/pid/)** ("Pixel Diffusion Decoder")
model, and it is engineered so the result **always comes back**, even when no GPU is available.

The whole front end is a static site on GitHub Pages. The default engine runs **entirely in your
browser**, so for most images you do not need a server or a GPU at all.

---

## ✨ Features

| | Feature |
|---|---|
| 🎚️ | **Three engines, you choose**: Fast and High Detail run in your browser; **NVIDIA PiD** runs live on a GPU |
| 🧠 | **NVIDIA PiD** diffusion upscaling on Modal, auto-disabled with a clear message if it is ever at capacity |
| ⚡ | The browser engines need **no account, no server, no cost**, and work offline |
| 🪟 | **Before / after slider** to inspect the result at a glance |
| 🫧 | **Transparency preserved** for PNGs (alpha is upscaled separately) |
| 💾 | **Download** as PNG or JPG, with the output file size shown |
| 📋 | Drag and drop, click to browse, **or paste an image** with Cmd/Ctrl+V |
| 🛑 | **Cancel** any run mid-process |
| 🔒 | **Zero storage**: images are processed in memory and never persisted |

---

## 🧠 How It Works

You pick the engine. Two run on your own machine, one runs NVIDIA PiD on a GPU:

| Engine | Where it runs | Notes |
|--------|---------------|-------|
| **Fast** | Browser, `realesr-general-x4v3` (4.9 MB) | Instant, great for most images, works offline |
| **High Detail** | Browser, `RealESRGAN_x4plus` (67 MB) | Sharper, heavier, best with WebGPU |
| **NVIDIA PiD** | GPU via [Modal](https://modal.com) (serverless) | The headline diffusion model, 4× to 2048px, cold start up to a minute |

The browser models are vendored same-origin in `public/models/` and lazy-loaded on first use, so
the site is fully functional with no backend at all. The **NVIDIA PiD** engine calls a CORS-enabled
Modal endpoint (no secret in the browser, auth stays server side). PiD internally tries Modal first,
then an optional [Hugging Face ZeroGPU Space](https://huggingface.co/docs/hub/spaces-zerogpu); if no
PiD backend can serve, or its rate limit is reached, the **PiD button disables itself with a "come
back later" message** and the upload finishes on the local Fast engine so you are never stranded.

> **Which should you use?** Fast for everyday images (instant, private, free). NVIDIA PiD when you
> want the strongest result and do not mind a short GPU wait.

---

## 🔒 Privacy

Processing is **stateless**. The image is sent (or processed locally), upscaled, and returned in a
single pass:

- GPU backends delete their temporary files immediately after each request.
- The browser holds the result only as an in-memory blob, revoked the moment you download or reset.
- There is no bucket, no database, and no logging of image bytes. Nothing to leak, nothing to clean up.

No API token or secret ever ships in the static bundle. Backend auth lives server side only.

---

## 🚀 Quick Start

```bash
git clone https://github.com/natekali/imgupscaler.git
cd imgupscaler
npm install
npm run dev      # http://localhost:5173/imgupscaler/
npm run build    # production build into dist/
```

The site is fully functional out of the box thanks to the in-browser engine. No keys, no backend
required to develop or self-host.

---

## 🛠️ Deploy

Pushing to `main` triggers `.github/workflows/deploy-pages.yml`, which builds the site and
publishes it to GitHub Pages. To host it yourself, enable Pages with "GitHub Actions" as the
source and push.

---

## The NVIDIA PiD engine

PiD runs on Modal (serverless GPU). The backend code lives in `modal/modal_app.py`: it clones
NVIDIA PiD, bakes the bundled weights (decoder + VAE, no gated FLUX needed) into the image, and
serves `GET /health` and `POST /upscale`. Deploy and wire it with:

| Backend | Command | Set in `src/config.ts` |
|---------|---------|------------------------|
| Modal (primary) | `modal deploy modal/modal_app.py` | `modalBase` (the printed `*.modal.run` base) |
| HF Space (optional secondary) | push `space/` to a new Space, Hardware = ZeroGPU | `hfSpace` |

No secret is placed in the front end. Modal and Hugging Face auth stay server side. Leave both
empty and the **NVIDIA PiD** option simply shows as unavailable while the browser engines keep
working.

---

## 🧰 Tech Stack

| Layer | Tech |
|-------|------|
| Front end | Vite, TypeScript, modern CSS (no framework) |
| In-browser inference | onnxruntime-web (WebGPU with WASM fallback), Real-ESRGAN x4 |
| GPU inference | NVIDIA PiD on Modal (FastAPI) and Hugging Face ZeroGPU (Gradio) |
| Compare slider | img-comparison-slider |
| Hosting | GitHub Pages (static), GitHub Actions deploy |

---

## 🙏 Credits

- Upscaling by **[NVIDIA PiD](https://github.com/nv-tlabs/PiD)** (Apache-2.0).
- In-browser fallback by **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)**.
- Runtime by **[ONNX Runtime Web](https://onnxruntime.ai)**.

---

## 📄 License

[Apache-2.0](./LICENSE). Built by [natekali](https://github.com/natekali).
