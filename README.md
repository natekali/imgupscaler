# Resolve — AI Image Upscaler

> Drop a low-resolution image, get a crisp 4K-grade result. Powered by NVIDIA's brand-new
> **[PiD](https://research.nvidia.com/labs/sil/projects/pid/)** pixel-diffusion decoder,
> with a resilient multi-engine backend that **always returns a result**.

A dead-simple, single-screen web app: upload → enhance → compare → download. The frontend
is a static site (GitHub Pages); the AI runs on GPU backends it calls cross-origin.

🔭 **Live:** https://natekali.github.io/imgupscaler/

## How it works — a resilient engine

Free GPU is scarce, so a single backend would inevitably show "busy" under load. Instead the
browser tries a **failover chain** and stops at the first engine that returns:

| Tier | Engine | Role |
|------|--------|------|
| **1** | NVIDIA **PiD** on [Modal](https://modal.com) (serverless GPU) | Reliable primary — free credits, CORS, secret stays server-side |
| **2** | NVIDIA **PiD** on a [HuggingFace ZeroGPU Space](https://huggingface.co/docs/hub/spaces-zerogpu) | Free, best-effort accelerator |
| **3** | **Real-ESRGAN x4** via [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (WebGPU→WASM) | In-browser floor — can never be "busy", works offline |

The in-browser tier (a 4.9 MB model vendored at `public/models/`) is the guarantee: even with
zero GPU backends alive, every upload still produces an upscaled image — at lower quality than
PiD, but it always works.

## Privacy — nothing is stored

Processing is **stateless**: the image is sent, upscaled, and returned in one request. GPU
backends delete their temp files immediately; the browser holds the result only as an
in-memory blob that's revoked the moment you download or reset. There is no bucket, no
database, nothing to leak or clean up.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173/imgupscaler/
npm run build    # → dist/
```

The site is fully functional out of the box via the in-browser engine. To enable the PiD GPU
tiers, deploy the backends and fill in their (non-secret) URLs in `src/config.ts`:

- **Modal (Tier 1):** `modal deploy modal/modal_app.py` → put the printed URL in `modalEndpoint`.
- **HF Space (Tier 2):** push `space/` to a new Space, set Hardware = ZeroGPU → put the Space
  id (e.g. `natekali/pid-upscaler`) in `hfSpace`.

> **No secrets in the frontend.** `src/config.ts` holds only public endpoint URLs. API tokens
> live server-side inside the Modal function / HF Space — never in this repo or the built site
> (a GitHub Pages bundle is world-readable).

## Deploy

Pushing to `main` triggers `.github/workflows/deploy-pages.yml`, which builds and publishes
to GitHub Pages.

## Credits

Upscaling by **[NVIDIA PiD](https://github.com/nv-tlabs/PiD)** (Apache-2.0). In-browser
fallback by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Built with Vite + TypeScript.

## License

[Apache-2.0](./LICENSE).
