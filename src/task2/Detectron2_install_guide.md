# Installing Detectron2 with PyTorch + CUDA (RTX 5090 / Blackwell)

> Tested on: Ubuntu 24.04, NVIDIA RTX 5090 (sm_120), Driver 570.211.01, CUDA 12.8

---

## Prerequisites

- NVIDIA GPU driver already installed (570.x+)
- Conda (Anaconda or Miniconda)

Verify your driver:

```bash
nvidia-smi
```

---

## Step 1 — Install CUDA Toolkit 12.8

The RTX 5090 (Blackwell, sm_120) requires CUDA 12.8. Your system `nvcc` must match.

### Option A: Deb package (recommended for Ubuntu 24.04)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### Option B: Runfile installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
```

> **Important:** During runfile install, **uncheck the driver** — only install the toolkit.

If neither URL works, get the exact link from:
https://developer.nvidia.com/cuda-12-8-0-download-archive

### Set environment variables

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

Apply changes:

```bash
source ~/.bashrc
```

### Verify

```bash
nvcc --version
# Expected: Cuda compilation tools, release 12.8, ...
```

---

## Step 2 — Create conda environment

```bash
conda create -n d2 python=3.11 -y
conda activate d2
```

---

## Step 3 — Install PyTorch with CUDA 12.8

> **Note:** PyTorch dropped conda install support from v2.6.0 onward. Use pip inside conda.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Verify PyTorch + GPU

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
from torch.utils.cpp_extension import CUDA_HOME
print('CUDA_HOME:', CUDA_HOME)
"
```

Expected output:

```
PyTorch version: 2.7.x+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
CUDA_HOME: /usr/local/cuda-12.8
```

---

## Step 4 — Install detectron2 dependencies

```bash
pip install ninja cython pycocotools
```

---

## Step 5 — Install detectron2

### Option A: Direct install from GitHub

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### Option B: Editable install (if you plan to modify detectron2)

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install --no-build-isolation -e .
```

> **`--no-build-isolation` is required** — without it, the build may fail to find the correct
> PyTorch/CUDA versions.

---

## Step 6 — Verify detectron2

**Important:** Run this from *outside* the detectron2 source directory.

```bash
cd ~
python -m detectron2.utils.collect_env
```

Check that these three values all show **12.8**:

- `Detectron2 CUDA Compiler`
- `CUDA_HOME`
- `PyTorch built with - CUDA`

Quick import test:

```bash
python -c "import detectron2; print('Detectron2 version:', detectron2.__version__)"
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Unsupported gpu architecture` during build | `TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e .` |
| `GLIBCXX` version errors | `conda install -c conda-forge libstdcxx-ng` |
| `ImportError: cannot import name '_C'` | Rebuild detectron2: `rm -rf build/ **/*.so` then reinstall |
| `nvcc not found` during build | Ensure `CUDA_HOME` is set and `nvcc` is on your `PATH` |
| PyTorch says `CUDA not available` | Reinstall PyTorch with the correct `cu128` index URL |
| Import errors when running from detectron2 dir | Always `cd` to a different directory before importing |