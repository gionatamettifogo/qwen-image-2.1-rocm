import json
import os
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
MODELS = Path(os.getenv("HF_HOME", "/models/huggingface"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
generation_lock = threading.Lock()

MODEL_FILES = {
    "diffusion": ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
    "vae": ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors"),
    "llm": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
    "llm_vision": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
}


def gpu_info():
    if not Path("/dev/kfd").exists() or not Path("/dev/dri/renderD128").exists():
        raise RuntimeError("ROCm compute devices are not exposed to the container")
    result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"rocminfo failed: {result.stderr.strip()}")
    match = re.search(r"Name:\s+(gfx\d+)", result.stdout)
    if not match:
        raise RuntimeError("No AMD GPU agent found in rocminfo")
    return {"gpu": "AMD Radeon 890M", "gfx": match.group(1), "hip": "ROCm 7.2"}


def ensure_models():
    paths = {}
    token = os.getenv("HF_TOKEN") or None
    started = time.perf_counter()
    for key, (repo, filename) in MODEL_FILES.items():
        paths[key] = hf_hub_download(
            repo_id=repo,
            filename=filename,
            token=token,
            cache_dir=str(MODELS),
        )
    return paths, time.perf_counter() - started


@asynccontextmanager
async def lifespan(_app):
    gpu_info()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Qwen Image 2.1 local", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        gpu = gpu_info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", **gpu, "backend": "stable-diffusion.cpp HIPBLAS"}


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(DEFAULT_STEPS),
    seed: int = Form(42),
    references: list[UploadFile] = File(default=[]),
):
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    if len(references) > 10:
        raise HTTPException(status_code=422, detail="at most 10 reference images")
    if width < 256 or height < 256 or width % 32 or height % 32:
        raise HTTPException(status_code=422, detail="width and height must be >=256 and divisible by 32")
    if not 1 <= steps <= 100:
        raise HTTPException(status_code=422, detail="steps must be between 1 and 100")

    with generation_lock:
        temps = []
        try:
            paths, download_seconds = ensure_models()
            ref_paths = []
            for upload in references:
                suffix = Path(upload.filename or "reference.png").suffix or ".png"
                tmp = NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(await upload.read())
                tmp.close()
                temps.append(tmp.name)
                ref_paths.append(tmp.name)

            request_id = uuid.uuid4().hex
            output_path = OUTPUTS / f"{request_id}.png"
            log_path = LOGS / f"{request_id}.log"
            command = [
                "sd-cli",
                "--diffusion-model", paths["diffusion"],
                "--vae", paths["vae"],
                "--llm", paths["llm"],
                "--llm_vision", paths["llm_vision"],
                "--backend", "te=cpu,vae=ROCm0,diffusion=ROCm0",
                "--sampling-method", "euler",
                "--cfg-scale", "6.0",
                "--steps", str(steps),
                "--width", str(width),
                "--height", str(height),
                "--seed", str(seed),
                "-p", prompt,
            ]
            for ref in ref_paths:
                command.extend(["-r", ref])
            command.extend(["-o", str(output_path)])

            started = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True)
            generation_seconds = time.perf_counter() - started
            log_path.write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"sd-cli exited {result.returncode}; see logs/{log_path.name}: "
                    f"{result.stderr[-1600:]}"
                )
            if not output_path.is_file():
                raise RuntimeError("sd-cli completed without creating an output image")

            gpu_state = gpu_info()
            record = {
                "id": request_id,
                "model": "Qwen/Qwen-Image-2.1 INT8 ConvRot",
                "text_encoder": "Qwen3-VL-8B-Instruct Q4_K_M",
                "gpu": gpu_state["gpu"],
                "gfx": gpu_state["gfx"],
                "backend": "HIPBLAS",
                "prompt": prompt,
                "reference_count": len(ref_paths),
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "model_download_seconds": round(download_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "output": f"outputs/{output_path.name}",
                "log": f"logs/{log_path.name}",
            }
            with (LOGS / "timings.jsonl").open("a") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            return record
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        finally:
            for path in temps:
                Path(path).unlink(missing_ok=True)


@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = OUTPUTS / filename
    if path.parent != OUTPUTS or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return {"path": f"outputs/{filename}"}
