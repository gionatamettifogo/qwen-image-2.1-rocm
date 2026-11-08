#!/usr/bin/env python3
"""End-to-end GPU tests: text-to-image and two-reference image conditioning."""
import json
import os
import struct
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = os.getenv("QWEN_URL", "http://127.0.0.1:8188")
TEST_STEPS = int(os.getenv("TEST_STEPS", "8"))
SIZE = int(os.getenv("TEST_SIZE", "512"))


def png_solid(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a small RGB PNG using only Python's standard library."""
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\0" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----qwen-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(),
            b"\r\n",
        ])
    for field, filename, data in files:
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            data,
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def generate(name: str, prompt: str, references=()) -> dict:
    fields = {
        "prompt": prompt,
        "width": str(SIZE),
        "height": str(SIZE),
        "steps": str(TEST_STEPS),
        "seed": "42",
    }
    files = [("references", filename, data) for filename, data in references]
    body, boundary = multipart(fields, files)
    request = urllib.request.Request(
        f"{BASE_URL}/generate",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
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
        "gfx": result.get("gfx", result.get("compute_capability")),
        "backend": result["backend"],
        "reference_count": result["reference_count"],
    }, indent=2))
    return result


def main():
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as response:
        health = json.load(response)
    if not health.get("gpu") or not health.get("backend"):
        raise RuntimeError(f"Expected active ROCm GPU, got {health}")

    generate(
        "text_to_image",
        "A tiny red fox reading a book beneath a brass reading lamp, "
        "cozy illustrated storybook, rich detail, square composition.",
    )
    references = [
        ("reference-red.png", png_solid(128, 128, (190, 45, 45))),
        ("reference-blue.png", png_solid(128, 128, (40, 95, 190))),
    ]
    generate(
        "two_reference_images",
        "Create a polished product photograph combining the red ceramic vase "
        "and the blue glass bowl from the two reference images on a pale stone table.",
        references,
    )


if __name__ == "__main__":
    main()
