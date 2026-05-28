"""
Upscaler AI, Modal serverless GPU backend (the NVIDIA PiD engine).

Runs NVIDIA PiD's official from-clean image upscaler on a Modal GPU and exposes a
CORS-enabled API the static GitHub Pages frontend calls directly:

  GET  /health   cheap liveness probe (the frontend uses it to enable/disable the PiD button)
  POST /upscale  multipart image in, upscaled PNG out

PiD is hosted as a Modal Cls so the model is loaded ONCE per container in @modal.enter()
and reused across requests. Warm requests skip the ~30 s model load that the CLI subprocess
path used to pay every call. If the in-process load fails for any reason, the endpoint
falls back to the original CLI subprocess so the service never goes fully dark.

Aspect ratio is preserved: we pad the input to a square with edge replication so PiD's
mandatory center-crop is a no-op, then crop the upscaled result back to the original aspect.

Deploy:   modal deploy modal/modal_app.py
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
DEFAULT_PROMPT = "a high quality, sharp, detailed photograph"
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
    """Pad the image to a square with edge replication so PiD's center-crop is a no-op."""
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


def _crop_to_aspect(pid_out_path: str, orig_w: int, orig_h: int,
                    pad: tuple[int, int, int, int]) -> bytes:
    """Crop PiD's square output back to the original aspect ratio (at the 4x scale)."""
    from PIL import Image

    pad_left, pad_right, pad_top, pad_bottom = pad
    out = Image.open(pid_out_path)
    out_w, _ = out.size
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


def _run_pid_cli(input_path: str, output_dir: str) -> str:
    """Slow CLI fallback. Used only when the in-process model failed to load."""
    cmd = [
        "python", "-m", "pid._src.inference.from_clean_flux",
        "--input_path", input_path,
        "--input_resolution", str(INPUT_RESOLUTION),
        "--degrade_sigmas", "0.0",
        "--output_dir", output_dir,
        "--cfg_scale", "1",
        "--pid_inference_steps", str(PID_STEPS),
        "--scale", str(SCALE),
        "--pid_ckpt_type", CKPT_TYPE,
        "--save_format", "png",
        "--prompt", DEFAULT_PROMPT,
    ]
    env = {**os.environ, "PYTHONPATH": PID_DIR}
    proc = subprocess.run(cmd, cwd=PID_DIR, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"PiD CLI exited {proc.returncode}:\n{(proc.stderr or proc.stdout or '')[-3000:]}")
        raise RuntimeError("PiD inference failed")
    candidates = [
        p
        for p in glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
        if "/vae_decode/" not in p and "/input/" not in p
    ]
    if not candidates:
        raise RuntimeError("PiD produced no output image")
    return max(candidates, key=os.path.getmtime)


@app.cls(image=image, gpu="A10G", timeout=600, scaledown_window=180)
class PiDService:
    """Loads PiD once per container; serves /upscale and /health via FastAPI."""

    @modal.enter()
    def load(self):
        """Load the PiD model into VRAM exactly once. Subsequent requests reuse it."""
        self.model = None
        try:
            import sys
            import torch  # noqa: F401  (imported here to fail fast at startup if missing)

            # PiD's checkpoint paths are relative to the repo root (e.g. "checkpoints/...").
            os.chdir(PID_DIR)
            if PID_DIR not in sys.path:
                sys.path.insert(0, PID_DIR)

            from pid._src.inference.checkpoint_registry import get_pid_checkpoint
            from pid._src.utils.model_loader import load_model_from_checkpoint

            ckpt = get_pid_checkpoint("flux", CKPT_TYPE)
            model, _config = load_model_from_checkpoint(
                experiment_name=ckpt.experiment,
                checkpoint_path=ckpt.checkpoint_path,
                config_file="pid/_src/configs/pid/config.py",
                enable_fsdp=False,
                experiment_opts=[],
                strict=False,
                load_ema_to_reg=False,
            )
            model.eval()
            self.model = model
            print("PiD in-process model loaded successfully (warm requests will be fast).")
        except Exception as e:
            import traceback
            print("PiD in-process load FAILED; will fall back to CLI subprocess per request.")
            print(f"  reason: {e}")
            traceback.print_exc()
            self.model = None

    def _upscale_in_process(self, input_path: str) -> "object":
        """Run PiD on a pre-padded square image and return a PIL Image (square, 4x)."""
        import numpy as np
        import torch
        from PIL import Image
        from pid._src.inference._demo_from_clean_common import _load_input_image, _vae_decode

        # Center-crop (no-op on our padded square) + bicubic-resize to 512 + scale to [-1, 1].
        input_tensor = _load_input_image(input_path, INPUT_RESOLUTION, keep_input_size=False).to(
            dtype=torch.bfloat16, device="cuda"
        )
        with torch.no_grad():
            clean_latent = self.model.encode_lq_latent(input_tensor)  # [1, C, zH, zW]
            vae_img = _vae_decode(self.model, clean_latent)  # [1, 3, R, R] in [-1, 1]
            lq_placeholder = torch.zeros_like(vae_img, dtype=torch.bfloat16, device="cuda")

            vae_compression = int(self.model.vae_encoder.spatial_compression_factor)
            vae_h = int(clean_latent.shape[-2]) * vae_compression
            vae_w = int(clean_latent.shape[-1]) * vae_compression
            target_hw = (vae_h * SCALE, vae_w * SCALE)

            data_batch = {
                self.model.config.input_caption_key: [DEFAULT_PROMPT],
                "LQ_video_or_image": lq_placeholder,
                "LQ_latent": clean_latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor([0.0], device="cuda", dtype=torch.float32),
            }
            samples = self.model.generate_samples_from_batch(
                data_batch,
                cfg_scale=1.0,
                num_steps=PID_STEPS,
                seed=5,
                shift=None,
                image_size=target_hw,
            )

        # samples[0] is [C, H, W] (or [C, 1, H, W] from some backbones). Convert to PIL.
        ours = samples[0].float().cpu().clamp(-1, 1)
        if ours.dim() == 4:
            ours = ours.squeeze(1)
        arr = ((ours.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @modal.asgi_app()
    def web(self):
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
            return JSONResponse({"ok": True, "engine": "nvidia-pid", "warm": self.model is not None})

        @api.post("/upscale")
        async def upscale(image: UploadFile = File(...)):  # noqa: B008
            work = tempfile.mkdtemp(prefix="pid_")
            try:
                in_path = os.path.join(work, "input.png")
                raw = await image.read()
                orig_w, orig_h, pad = _pad_to_square(raw, in_path)

                if self.model is not None:
                    # Fast path: model already in VRAM.
                    pid_pil = self._upscale_in_process(in_path)
                    pid_full = os.path.join(work, "pid_full.png")
                    pid_pil.save(pid_full, format="PNG", optimize=False)
                else:
                    # Safe fallback: reload model from disk every call (slow but correct).
                    pid_full = _run_pid_cli(in_path, os.path.join(work, "out"))

                data = _crop_to_aspect(pid_full, orig_w, orig_h, pad)
                return Response(content=data, media_type="image/png")
            except Exception as e:
                import traceback
                print("PiD upscale failed:", e)
                traceback.print_exc()
                return Response(
                    content="Upscaling failed. Please try again later.",
                    media_type="text/plain",
                    status_code=500,
                )
            finally:
                shutil.rmtree(work, ignore_errors=True)

        return api
