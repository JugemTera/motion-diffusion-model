#!/usr/bin/env bash
# Build the MDM (motion-diffusion-model) Docker image.
set -euo pipefail

# The docker-shell directory itself is the build context: the Dockerfile copies nothing
# from the repo, so the (large) repo root is never sent to the Docker daemon.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-mdm:cu121}"

echo "Building ${IMAGE_NAME} (context: ${SCRIPT_DIR})"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo "Done. Image: ${IMAGE_NAME}"
