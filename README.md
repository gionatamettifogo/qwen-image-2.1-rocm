# Qwen Image 2.1 on Seratonin

A headless, SSH-operated image-generation service for the AMD Radeon 890M integrated GPU in the Minisforum AI X1 Pro. The container builds upstream stable-diffusion.cpp with HIPBLAS for gfx1150 and refuses to start without an AMD GPU visible through ROCm. A display is not required.

The diffusion transformer uses the INT8 ConvRot Qwen Image 2.1 checkpoint, with a BF16 Qwen Image 2.1 VAE and a Q4_K_M Qwen3-VL 8B GGUF text/vision encoder. The runner uses CPU offload with the host's unified memory.

## Tested systems and acceleration stack

| Host | OS and kernel | GPU and host driver | Container and acceleration |
|---|---|---|---|
| Seratonin, this repo | Ubuntu 24.04.5 LTS; Linux 7.0.0-38-generic | AMD Radeon 890M, gfx1150, integrated into Ryzen AI 9 HX370. rocm-smi driver report: 7.0.0-38-generic (same as the host kernel release). | Docker 29.8.0; ROCm 7.2.1 PyTorch image on Ubuntu 24.04 with Python 3.12 and PyTorch 2.9.1; HIPBLAS build for gfx1150. |
| Sparky, NVIDIA comparison | Ubuntu 24.04.5 LTS; Linux 7.0.0-1019-nvidia | NVIDIA GB10, compute capability 12.1; driver 595.99.02. NVIDIA-SMI reports CUDA 13.2 support. | Docker 29.6.2; NVIDIA CUDA 12.9.1 Ubuntu 24.04 image; stable-diffusion.cpp compiled for sm_121. |

Seratonin is a Minisforum AI X1 Pro with AMD Ryzen AI 9 HX370 (12 cores / 24 threads, up to 5.1 GHz), Radeon 890M, 96 GB DDR5, and a 2 TB PCIe 4.0 SSD. Docker exposes the GPU to the container through /dev/kfd and /dev/dri. The ROCm 7.2.1 user-space stack is in the container; the host supplies the Linux kernel and GPU device interface.

## Matched test results

Both hosts used the same INT8 ConvRot diffusion checkpoint, BF16 VAE, Q4_K_M text/vision encoder, prompts, reference images, seeds, 768×1024 output size, and 20 steps.

| Test | Seratonin, AMD Radeon 890M | Sparky, NVIDIA GB10 |
|---|---:|---:|
| Unicorn text-to-image, seed 28417 | 288.438 s | 32.765 s |
| Woman-at-harbor with two reference images, seed 73023 | 566.506 s | 165.620 s |

These are local generation times, including per-request model loading, and excluding model download and image upload. Seratonin used its local model cache for the recorded tests. On Sparky, the first model download took 198.539 s for approximately 14 GB; subsequent requests used its local cache.

## Example outputs from Seratonin

These committed PNGs are the outputs from the two matched tests. The reference-conditioned test uses the portrait and harbor images in tests/assets/.

<table>
  <tr><th>Text prompt · seed 28417 · 288.438 s</th><th>Two reference images · seed 73023 · 566.506 s</th></tr>
  <tr>
    <td><img src="tests/results/unicorn-riding-harley.png" alt="Ivory unicorn riding a black Harley on a mountain road at sunset" width="360"></td>
    <td><img src="tests/results/woman-sailing-at-harbor.png" alt="Woman from the portrait reference preparing a sailboat at the harbor" width="360"></td>
  </tr>
</table>

The sample files are tracked under tests/results/ and arrive with git pull. The exact prompts, seeds, and input images are in tests/generate_examples.py and tests/assets/.

## Install, start, and stop

Run these commands over SSH:

    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./install.sh'
    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./start.sh'
    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && ./stop.sh'

For Hugging Face access, copy .env.example to .env, accept the model licenses if requested, and set HF_TOKEN in that ignored local file. The first generation downloads approximately 14 GB of model weights into models/.

The API binds to 127.0.0.1:8188. From a Mac, use an SSH tunnel:

    ssh -N -L 8188:127.0.0.1:8188 seratonin

## Reproduce the tests

With the service running:

    ssh seratonin 'cd ~/github/qwen-image-2.1-rocm && python3 tests/generate_examples.py'

The script prints output paths and timings. Fresh PNGs go under outputs/; request timings are appended to logs/timings.jsonl, and native inference logs are saved under logs/. To refresh the committed samples, copy the newly generated matching PNGs into tests/results/ using the names above, then commit the updated images. The tests default to 768×1024 and 20 steps; set TEST_WIDTH, TEST_HEIGHT, or TEST_STEPS to change those settings.

generation_seconds measures local model loading and inference, excluding initial weight downloads and uploads. model_download_seconds records the download time separately.
