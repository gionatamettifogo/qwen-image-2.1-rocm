# Qwen Image 2.1 on Seratonin

A headless, SSH-operated image-generation service for Seratonin's Radeon 890M.
The container builds the upstream stable-diffusion.cpp HIPBLAS backend and
refuses to start without an AMD GPU visible through ROCm. A display is not
required.

The diffusion transformer uses the INT8 ConvRot checkpoint from
Comfy-Org/Qwen-Image-2.1, with the BF16 Qwen Image 2.1 VAE and a Q4_K_M
Qwen3-VL 8B GGUF text/vision encoder. This keeps the main model quantized while
retaining an 8-bit quality-focused weight format. The inference runner uses
CPU offload to work within the Radeon 890M's reserved VRAM while sharing the
host's unified memory. Image generation runs locally after the weights
download.

## Install, start and stop

Run commands over SSH:

    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./install.sh'
    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./start.sh'
    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./stop.sh'

For Hugging Face access, optionally copy .env.example to .env, accept the
model licenses if requested, and add a read token as HF_TOKEN. The first
generation downloads approximately 14 GB of model weights into models/.

The API binds to 127.0.0.1:8188. From a Mac, use an SSH tunnel:

    ssh -N -L 8188:127.0.0.1:8188 seratonin

Then call http://127.0.0.1:8188/generate using multipart form fields:
prompt (required), repeated references image files (up to 10), and optional
width, height, steps, and seed. Output PNGs are saved under outputs/.

## Smoke tests

With the service running, generate one text-only image and another using two
reference images:

    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && python3 tests/generate_examples.py'

The text and reference tests use the same prompts and seeds for host-to-host
comparison. They default to 768×1024 and 20 steps. Set TEST_WIDTH,
TEST_HEIGHT, or TEST_STEPS to adjust the settings. Reference images are in
tests/assets/.

## Timing and logs

Each request writes a JSON line to logs/timings.jsonl and a full native
inference log to logs/<request-id>.log. generation_seconds measures the
entire local inference process, including per-request model loading, but not
the initial weight download or upload. model_download_seconds records the
download time separately.
