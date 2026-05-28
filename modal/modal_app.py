"""
Upscaler AI, Modal serverless GPU backend (the NVIDIA PiD engine).

Runs NVIDIA PiD's official from-clean image upscaler on a Modal GPU and exposes a
CORS-enabled API the static GitHub Pages frontend calls directly:

  GET  /health   cheap liveness probe (the frontend uses it to enable/disable the PiD button)
  POST /upscale  multipart image in, upscaled PNG out

Why it works without gated weights: nvidia/PiD ships its own VAE (checkpoints/ae.safetensors),
and the from-clean path decodes via the model's bundled VAE, so FLUX.1-dev (gated) is never
downloaded. The PiD repo is cloned and `pip install -e`'d so the relative `checkpoints/...`
paths in its checkpoint registry resolve at runtime.

Deploy:   modal deploy modal/modal_app.py
Then put the printed `/upscale` URL in src/config.ts as `modalEndpoint`.
"""

import glob
import io
import os
import shutil
import subprocess
import tempfile

import modal

PID_DIR = "/pid"  # cloned repo + downloaded weights live here (cwd at runtime)
INPUT_RESOLUTION = 512
SCALE = 4
PID_STEPS = 4
CKPT_TYPE = "2k"  # 512 -> 2048 (4x) decoder for the Flux backbone
ALLOWED_ORIGINS = ["https://natekali.github.io", "http://localhost:5173"]

app = modal.App("upscaler-ai-pid")

# Build once (cached): clone PiD, install it + deps, and bake the nvidia/PiD weights
# (decoder + bundled VAE) into the image at /pid/checkpoints so nothing downloads per request.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers>=0.37",
        "transformers>=4.57",
        "accelerate",
        "safetensors",
        "pillow",
        "huggingface_hub",
        "fastapi[standard]",
        # PiD's own runtime dependencies (not all are pulled by `pip install -e`).
        "hydra-core",
        "omegaconf",
        "pyyaml",
        "attrs",
        "einops",
        "loguru",
        "termcolor",
        "fvcore",
        "iopath",
        "wandb",
        "imageio",
        "opencv-python-headless",
        "pandas",
        "sentencepiece",
        "boto3",
        "botocore",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/nv-tlabs/PiD.git {PID_DIR}",
        f"pip install -e {PID_DIR}",
        # Bundled weights (decoder + ae.safetensors VAE) into /pid/checkpoints.
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='nvidia/PiD', local_dir='{PID_DIR}', "
        "allow_patterns=['checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*', "
        "'checkpoints/ae.safetensors', 'config.json'])\"",
    )
)


def _pad_to_square(img_bytes: bytes, save_to: str) -> tuple[int, int, tuple[int, int, int, int]]:
    """Pad the image to a square with edge replication so PiD's center-crop is a no-op.

    Returns (orig_W, orig_H, (pad_left, pad_right, pad_top, pad_bottom)). PiD's from-clean
    pipeline always center-crops + resizes to a square; if we feed it a non-square image we
    permanently lose the side bands. Padding preserves the full image, and we crop the
    upscaled output back to the original aspect afterwards.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    side = max(W, H)
    pad_left = (side - W) // 2
    pad_right = side - W - pad_left
    pad_top = (side - H) // 2
    pad_bottom = side - H - pad_top
    if pad_left or pad_right or pad_top or pad_bottom:
        arr = np.array(img)
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        img = Image.fromarray(padded)
    img.save(save_to, format="PNG", optimize=False)
    return W, H, (pad_left, pad_right, pad_top, pad_bottom)


def _crop_to_aspect(
    pid_out_path: str,
    orig_w: int,
    orig_h: int,
    pad: tuple[int, int, int, int],
) -> bytes:
    """Crop the PiD square output back to the original aspect ratio (at the 4x scale)."""
    from PIL import Image

    pad_left, pad_right, pad_top, pad_bottom = pad
    out = Image.open(pid_out_path)
    out_w, _ = out.size  # PiD output is square
    side = orig_w + pad_left + pad_right
    scale = out_w / side
    left = int(round(pad_left * scale))
    top = int(round(pad_top * scale))
    right = int(round((pad_left + orig_w) * scale))
    bottom = int(round((pad_top + orig_h) * scale))
    cropped = out.crop((left, top, right, bottom))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _run_pid(input_path: str, output_dir: str) -> str:
    """Invoke PiD's from-clean Flux upscaler and return the path of the 'ours' (PiD) output."""
    cmd = [
        "python", "-m", "pid._src.inference.from_clean_flux",
        "--input_path", input_path,
        "--input_resolution", str(INPUT_RESOLUTION),
        "--degrade_sigmas", "0.0",          # clean round trip, no synthetic degradation
        "--output_dir", output_dir,
        "--cfg_scale", "1",                 # no classifier-free guidance
        "--pid_inference_steps", str(PID_STEPS),
        "--scale", str(SCALE),
        "--pid_ckpt_type", CKPT_TYPE,
        "--save_format", "png",
        "--prompt", "a high quality, sharp, detailed photograph",
    ]
    env = {**os.environ, "PYTHONPATH": PID_DIR}
    proc = subprocess.run(cmd, cwd=PID_DIR, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        # Log the detail server-side; never return internals to the browser.
        print(f"PiD exited {proc.returncode}:\n{(proc.stderr or proc.stdout or '')[-3000:]}")
        raise RuntimeError("PiD inference failed")

    # PiD writes the VAE baseline under .../vae_decode/... and the input copy under .../input/...
    # The PiD ("ours") result is the remaining image under <tag>/sigma_0.000/<name>.png.
    candidates = [
        p
        for p in glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
        if "/vae_decode/" not in p and "/input/" not in p
    ]
    if not candidates:
        raise RuntimeError("PiD produced no output image")
    return max(candidates, key=os.path.getmtime)


@app.function(image=image, gpu="A10G", timeout=600, scaledown_window=180)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    api = FastAPI()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health():
        return JSONResponse({"ok": True, "engine": "nvidia-pid"})

    @api.post("/upscale")
    async def upscale(image: UploadFile = File(...)):  # noqa: B008
        work = tempfile.mkdtemp(prefix="pid_")
        try:
            in_path = os.path.join(work, "input.png")
            raw = await image.read()
            # Pad to square (edge-replicate) so PiD's mandatory center-crop is a no-op
            # and no content is lost; remember the padding so we can crop the output back.
            orig_w, orig_h, pad = _pad_to_square(raw, in_path)
            pid_out_path = _run_pid(in_path, os.path.join(work, "out"))
            data = _crop_to_aspect(pid_out_path, orig_w, orig_h, pad)
            return Response(content=data, media_type="image/png")
        except Exception as e:
            print("PiD upscale failed:", e)
            return Response(
                content="Upscaling failed. Please try again later.",
                media_type="text/plain",
                status_code=500,
            )
        finally:
            # Stateless: nothing is persisted between requests.
            shutil.rmtree(work, ignore_errors=True)

    return api
