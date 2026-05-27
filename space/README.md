---
title: Resolve PiD Upscaler
emoji: 🔭
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 4x image super-resolution powered by NVIDIA PiD
---

# Resolve · NVIDIA PiD Upscaler (Space backend)

This Space is the free, best-effort GPU backend (Tier 2) for the
[Resolve](https://github.com/natekali/imgupscaler) image upscaler. It wraps NVIDIA's
[PiD](https://github.com/nv-tlabs/PiD) pixel-diffusion decoder and exposes an `/upscale`
API endpoint that the static frontend calls anonymously via `@gradio/client`.

## Setup

1. Create the Space, push `app.py` + `requirements.txt` + this `README.md`.
2. In **Settings → Hardware**, select **ZeroGPU** (PiD needs a CUDA GPU; ZeroGPU is
   Gradio-only, which is why this backend uses the Gradio SDK).
3. First request cold-starts while weights load into VRAM; later calls are fast.

No secrets live here that the frontend can see — the frontend never sends a token, so this
backend draws on HF's shared anonymous GPU quota. When that's exhausted, the website simply
falls over to its next engine, so users are never blocked.
