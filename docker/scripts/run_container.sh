#!/bin/bash

# Run the BDD Detection Bench Docker container

# Source environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/.bdd_env_variables"

IMAGE_NAME="bdd-detection-bench"
IMAGE_TAG="v${DOCKER_IMAGE_VERSION}"
CONTAINER_NAME="${IMAGE_NAME}-${USER}"

# Get absolute path of the project root (parent of docker/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to cleanup existing container
cleanup_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping and removing existing container '${CONTAINER_NAME}'..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

# Function to start container
start_container() {
    echo "========================================"
    echo "Starting container: ${CONTAINER_NAME}"
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "========================================"
    
    # Cleanup if exists
    cleanup_container
    
    # Create container with volume mounts
    docker run -dit \
        --name "${CONTAINER_NAME}" \
        --hostname bdd-container \
        -v "${PROJECT_ROOT}:/workspace" \
        -w /workspace \
        --user "$(id -u):$(id -g)" \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Container started successfully!"
        echo "========================================"
        echo ""
        echo "To enter the container, run one of:"
        echo "  docker exec -it ${CONTAINER_NAME} bash"
        echo "  Or use: docker-run (if you sourced .bdd_aliases)"
        echo ""
        echo "To stop: docker-stop or 'docker stop ${CONTAINER_NAME}'"
        echo "========================================"
    else
        echo "ERROR: Failed to start container"
        exit 1
    fi
}

# Function to exec into running container
exec_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' is not running."
        echo "Starting it now..."
        start_container
        echo ""
    fi
    
    echo "Entering container '${CONTAINER_NAME}'..."
    docker exec -it "${CONTAINER_NAME}" bash
}

# Main logic
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container is running, exec into it
    exec_container
else
    # Container doesn't exist or not running, start it
    start_container
fi
