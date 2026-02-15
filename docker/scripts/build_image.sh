#!/bin/bash

# Build the Docker image for BDD Detection Bench

# Source environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/.bdd_env_variables"

IMAGE_NAME="bdd-detection-bench"
IMAGE_TAG="v${DOCKER_IMAGE_VERSION}"

echo "========================================"
echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "========================================"

# Build the image from the docker directory
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} "$SCRIPT_DIR"

# Tag as latest
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest

echo ""
echo "========================================"
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Also tagged as: ${IMAGE_NAME}:latest"
echo "========================================"
