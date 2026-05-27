"""
Resolve — Modal serverless GPU backend (Tier 1, reliable primary).

Runs NVIDIA PiD's `from_clean_flux` upscaler on a Modal GPU and exposes a CORS-enabled
`POST /upscale` endpoint that the static GitHub Pages frontend calls directly. Modal's
free monthly credits (~$30) cover on the order of a few thousand upscales/month, with
proper request attribution — this is what makes "always returns a result" true at the GPU
tier, since no secret ever has to live in the public browser bundle (Modal auth is
server-side).

Deploy:   modal deploy modal/modal_app.py
The deploy prints the public URL — put it in src/config.ts as `modalEndpoint`.

────────────────────────────────────────────────────────────────────────────────────────
DEPLOY NOTES (validate once on the GPU):
  • Confirm CLI flags:  python -m pid._src.inference.from_clean_flux --help
  • FLUX.1-dev components are fine for the user's free/personal use.
  • Tighten `allow_origins` to the final Pages origin before going public.
────────────────────────────────────────────────────────────────────────────────────────
"""

import glob
import os
import shutil
import subprocess
import tempfile

import modal

INPUT_RESOLUTION = 512
SCALE = 4
PID_STEPS = 4
CKPT_TYPE = "2kto4k"
ALLOWED_ORIGINS = ["https://natekali.github.io", "http://localhost:5173"]

app = modal.App("resolve-pid-upscaler")

# Build the image once: install PiD (brings diffusers/transformers/etc.) and bake the
# weights into the image so containers start without re-downloading.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "pillow", "fastapi[standard]", "huggingface_hub")
    .pip_install("git+https://github.com/nv-tlabs/PiD.git")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='nvidia/PiD')\""
    )
)


def _run_pid(input_path: str, output_dir: str) -> str:
    cmd = [
        "python", "-m", "pid._src.inference.from_clean_flux",
        "--input_path", input_path,
        "--input_resolution", str(INPUT_RESOLUTION),
        "--output_dir", output_dir,
        "--pid_inference_steps", str(PID_STEPS),
        "--scale", str(SCALE),
        "--pid_ckpt_type", CKPT_TYPE,
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})
    produced = sorted(
        glob.glob(os.path.join(output_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(output_dir, "**", "*.jpg"), recursive=True),
        key=os.path.getmtime,
    )
    if not produced:
        raise RuntimeError("PiD produced no output image")
    return produced[-1]


@app.function(image=image, gpu="A10G", timeout=300, scaledown_window=120)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    api = FastAPI()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.post("/upscale")
    async def upscale(image: UploadFile = File(...)):  # noqa: B008
        work = tempfile.mkdtemp(prefix="pid_")
        try:
            in_path = os.path.join(work, "input.png")
            with open(in_path, "wb") as f:
                f.write(await image.read())
            out_path = _run_pid(in_path, os.path.join(work, "out"))
            with open(out_path, "rb") as f:
                data = f.read()
            return Response(content=data, media_type="image/png")
        finally:
            # Stateless: nothing is persisted between requests.
            shutil.rmtree(work, ignore_errors=True)

    return api
