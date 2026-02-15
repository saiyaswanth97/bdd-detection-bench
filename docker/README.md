# BDD Detection Bench Docker Setup

Docker-based environment for running BDD100K detection benchmark task 1. The container mounts your entire project at `/workspace`, so any changes you make inside the container are reflected on your host system and vice versa.

## Quick Start

### 1. Source Aliases (One-time per session)

```bash
source docker/.bdd_aliases
```

### 2. Build the Docker Image

```bash
docker-build  # or: bash docker/scripts/build_image.sh
```

This creates a Docker image with all dependencies pre-installed:
- Python 3.11
- numpy, matplotlib, tqdm

### 3. Run the Container

```bash
docker-run  # or: dr
```

This will:
- Start the container if it's not running
- Or attach to the container if already running
- Mount your project directory to `/workspace`

### 4. Inside the Container

Once inside, you can run your tasks:

```bash
# Run task1 scripts
python3 -m src.task1.class_distribution
python3 -m src.task1.anamoly_detection
python3 -m src.task1.bbox_distribution
```

### 5. Stop the Container

```bash
docker-stop  # or: ds
```

---

## Configuration

### Environment Variables

Edit `docker/.bdd_env_variables` to configure:

```bash
DOCKER_IMAGE_VERSION=1  # Increment when you rebuild
BDD_DATA_PATH="data/bdd100k_labels"  # Dataset location
```

### Adding Python Dependencies

1. Add package to `docker/requirements.txt`
2. Increment `DOCKER_IMAGE_VERSION` in `.bdd_env_variables`
3. Rebuild: `docker-build`

---