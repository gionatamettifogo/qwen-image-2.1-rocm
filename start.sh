#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable or the current user cannot access it." >&2
  exit 1
fi
if ! docker image inspect qwen-image-2.1-rocm:local >/dev/null 2>&1; then
  DOCKER_BUILDKIT=0 docker compose build
fi
docker compose up -d
for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${PORT:-8188}/health" >/dev/null 2>&1; then
    echo "Qwen Image is ready at http://127.0.0.1:${PORT:-8188}"
    echo "Generate with: python3 tests/generate_examples.py"
    exit 0
  fi
  if [[ "$(docker inspect -f {{.State.Running}} qwen-image-2.1 2>/dev/null || true)" != true ]]; then
    docker compose logs --tail=100 >&2
    exit 1
  fi
  sleep 1
done
echo "Service did not become ready; see: docker compose logs -f" >&2
exit 1
