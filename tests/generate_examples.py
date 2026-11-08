"""Reproduce the same text and two-reference Qwen Image 2.1 cases on each host."""
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = os.getenv("QWEN_URL", "http://127.0.0.1:8188")
WIDTH = int(os.getenv("TEST_WIDTH", "768"))
HEIGHT = int(os.getenv("TEST_HEIGHT", "1024"))
STEPS = int(os.getenv("TEST_STEPS", "20"))
ASSETS = Path(__file__).resolve().parent / "assets"

UNICORN_PROMPT = (
    "A beautifully detailed cinematic fantasy illustration of a majestic ivory-white unicorn riding a classic black "
    "Harley-Davidson cruiser motorcycle along an open mountain road at golden hour. The unicorn is naturally seated "
    "astride the motorcycle, front hooves holding the handlebars, its long iridescent silver and pastel mane streaming "
    "in the wind, a small tasteful leather riding jacket, clearly visible single spiraled horn, expressive confident "
    "face. Three-quarter front view, motorcycle fully visible, realistic chrome and leather details, warm sunset light, "
    "dramatic alpine scenery, subtle road motion blur, whimsical yet elegant, premium storybook concept art, rich "
    "painterly realism, finely composed portrait image. No extra limbs, no duplicate horn, no words, no lettering, no watermark."
)
REFERENCE_PROMPT = (
    "Create a new photorealistic cinematic scene at the exact harbor shown in the environment reference. Show the same "
    "adult woman from the portrait reference, preserving her warm olive skin, freckles across her nose, dark brown eyes, "
    "short jaw-length curly dark-brown hair, cobalt blue knitted scarf, and mustard yellow waxed-cotton field jacket. "
    "She is now in a different situation: standing on the weathered wooden pier and preparing to sail, pulling a "
    "cream-colored sail upright on the small wooden boat tied alongside. Natural candid pose, three-quarter view, full "
    "torso and hands visible, focused optimistic expression. Keep the distinctive teal boathouse with its red door and "
    "the white lighthouse with navy cap clearly visible in the background, along with calm slate-blue water and the low "
    "green headland. Soft late-afternoon golden light, coherent reflections, high-end natural documentary photography, "
    "believable anatomy and hands, realistic proportions, no extra people, no text, no watermark."
)


def multipart(fields, files):
    boundary = "----qwen-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode(), b"\r\n",
        ])
    for field, filename, data in files:
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n", data, b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def generate(name, prompt, seed, references=()):
    fields = {"prompt": prompt, "width": WIDTH, "height": HEIGHT, "steps": STEPS, "seed": seed}
    body, boundary = multipart(fields, references)
    request = urllib.request.Request(
        f"{BASE_URL}/generate", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60 * 60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{name} failed: HTTP {exc.code}: {exc.read().decode()}") from exc
    output = Path(__file__).resolve().parents[1] / result["output"]
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{name} returned no image at {output}")
    print(json.dumps({
        "test": name,
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "generation_seconds": result["generation_seconds"],
        "model_download_seconds": result["model_download_seconds"],
        "gpu": result["gpu"],
        "architecture": result.get("gfx", result.get("compute_capability")),
        "backend": result["backend"],
        "reference_count": result["reference_count"],
        "width": result["width"], "height": result["height"],
        "steps": result["steps"], "seed": result["seed"],
    }, indent=2))


def main():
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as response:
        health = json.load(response)
    if not health.get("gpu") or not health.get("backend"):
        raise RuntimeError(f"Expected an active GPU backend, got {health}")
    generate("text_to_image_unicorn", UNICORN_PROMPT, 28417)
    refs = [
        ("references", "person-portrait.png", (ASSETS / "person-portrait.png").read_bytes()),
        ("references", "harbor-environment.png", (ASSETS / "harbor-environment.png").read_bytes()),
    ]
    generate("two_reference_images_harbor_scene", REFERENCE_PROMPT, 73023, refs)


if __name__ == "__main__":
    main()
