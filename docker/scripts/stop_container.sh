#!/bin/bash

# Stop the BDD Detection Bench Docker container

# Source environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/.bdd_env_variables"

IMAGE_NAME="bdd-detection-bench"
CONTAINER_NAME="${IMAGE_NAME}-${USER}"

echo "========================================"
echo "Stopping container: ${CONTAINER_NAME}"
echo "========================================"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
    echo ""
    echo "Container stopped and removed successfully."
else
    echo "Container '${CONTAINER_NAME}' not found."
fi

echo "========================================"
