"""
Upscaler AI, Hugging Face ZeroGPU Space (Tier 2 backend).

Wraps NVIDIA PiD's official image-upscaling entrypoint (`from_clean_flux`) in a Gradio
app with a `/upscale` API endpoint so the static frontend can call it via @gradio/client.

This is the FREE, best-effort accelerator in the failover chain. If ZeroGPU quota is
exhausted the frontend simply falls through to the next engine, so users never see a hard
error here.

────────────────────────────────────────────────────────────────────────────────────────
DEPLOY NOTES (validate once on the GPU, these can't be tested without CUDA):
  1. Confirm the exact CLI flags:  python -m pid._src.inference.from_clean_flux --help
     (arg names like --input_path / --output_dir / --pid_ckpt_type are from the README;
      adjust below if the repo differs.)
  2. PiD pulls FLUX components. User chose free/personal use, so FLUX.1-dev is fine.
  3. In the Space UI, set Hardware = ZeroGPU. Keep `sdk: gradio` (ZeroGPU is Gradio-only).
  4. First call cold-starts (weights → VRAM); subsequent calls are fast.
────────────────────────────────────────────────────────────────────────────────────────
"""

import glob
import os
import shutil
import subprocess
import tempfile

import gradio as gr
import spaces
from huggingface_hub import snapshot_download
from PIL import Image

# PiD upscaling parameters (from the official README).
INPUT_RESOLUTION = 512   # base latent resolution PiD encodes the clean image at
SCALE = 4                # 4× upscale
PID_STEPS = 4            # distilled (DMD2) decoder runs in 4 steps
CKPT_TYPE = "2kto4k"     # decoder variant that targets up to 4K output

# Pre-fetch PiD weights at startup so the first request doesn't pay the download cost.
WEIGHTS_DIR = snapshot_download(repo_id="nvidia/PiD")


def _run_pid(input_path: str, output_dir: str) -> str:
    """Invoke PiD's documented CLI and return the path of the produced image."""
    cmd = [
        "python", "-m", "pid._src.inference.from_clean_flux",
        "--input_path", input_path,
        "--input_resolution", str(INPUT_RESOLUTION),
        "--output_dir", output_dir,
        "--pid_inference_steps", str(PID_STEPS),
        "--scale", str(SCALE),
        "--pid_ckpt_type", CKPT_TYPE,
    ]
    env = {**os.environ, "PYTHONPATH": ".", "PID_WEIGHTS_DIR": WEIGHTS_DIR}
    subprocess.run(cmd, check=True, env=env)

    produced = sorted(
        glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(output_dir, "**", "*.jpg"), recursive=True),
        key=os.path.getmtime,
    )
    if not produced:
        raise RuntimeError("PiD produced no output image")
    return produced[-1]


@spaces.GPU(duration=120)
def upscale(image: Image.Image) -> Image.Image:
    """Upscale a clean RGB image 4× with NVIDIA PiD. Stateless: temp files are deleted."""
    work = tempfile.mkdtemp(prefix="pid_")
    try:
        in_path = os.path.join(work, "input.png")
        image.convert("RGB").save(in_path)
        out_path = _run_pid(in_path, os.path.join(work, "out"))
        # Load into memory before the temp dir is removed.
        return Image.open(out_path).copy()
    finally:
        shutil.rmtree(work, ignore_errors=True)


demo = gr.Interface(
    fn=upscale,
    inputs=gr.Image(type="pil", label="Low-resolution image"),
    outputs=gr.Image(type="pil", label="Upscaled 4×"),
    title="Upscaler AI · NVIDIA PiD Upscaler",
    description="4× super-resolution powered by NVIDIA's PiD pixel-diffusion decoder.",
    flagging_mode="never",
    api_name="upscale",
)

if __name__ == "__main__":
    demo.queue().launch()
