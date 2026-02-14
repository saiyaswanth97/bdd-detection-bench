# RT-DETR Training Setup Guide for BDD100K

This guide provides step-by-step instructions to set up RT-DETR training on BDD100K dataset on a new machine.

## Prerequisites

- NVIDIA GPU with CUDA support (minimum 6GB VRAM, 8GB+ recommended)
- Linux operating system (Ubuntu 18.04+)
- Python 3.8+
- Git

## 1. System Setup

### Install CUDA and cuDNN
```bash
# Check if CUDA is installed
nvidia-smi

# If not installed, follow NVIDIA's installation guide for your system
# https://developer.nvidia.com/cuda-downloads
```

### Install Python and pip
```bash
sudo apt update
sudo apt install python3.8 python3.8-dev python3-pip
```

## 2. Clone Repository

```bash
# Clone your detection benchmark repository
git clone <your-repo-url>
cd bdd-detection-bench
```

## 3. Install Python Dependencies

### Install PyTorch (Choose version based on your CUDA version)

For CUDA 11.8:
```bash
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Install RT-DETR Requirements
```bash
cd src/task2a2/RT-DETR/rtdetrv2_pytorch
pip3 install -r requirements.txt
```

**Required packages:**
- torch>=2.0.1
- torchvision>=0.15.2
- faster-coco-eval>=1.6.6
- PyYAML
- tensorboard
- scipy
- pycocotools
- onnx
- onnxruntime-gpu

## 4. Prepare BDD100K Dataset

### Download BDD100K

1. Register and download from: https://bdd-data.berkeley.edu/
2. Download the following files:
   - `bdd100k_images_100k.zip` (Images)
   - `bdd100k_labels.zip` (Annotations)

### Extract and Organize Dataset

```bash
# Create data directory at project root
cd /path/to/bdd-detection-bench
mkdir -p data

# Extract images
unzip bdd100k_images_100k.zip -d data/
# Expected structure: data/bdd100k_images_100k/100k/train/, val/, test/

# Extract labels
unzip bdd100k_labels.zip -d data/
# Expected structure: data/bdd100k_labels/100k/train/, val/, test/
```

### Convert BDD Annotations to COCO Format

```bash
# Run the conversion script (if available in your repo)
cd /path/to/bdd-detection-bench
python3 -m src.task2.convert_data

# This should create:
# data/bdd100k_labels/100k/train.json
# data/bdd100k_labels/100k/val.json
# data/bdd100k_labels/100k/test.json
```

**Expected Final Directory Structure:**
```
bdd-detection-bench/
├── data/
│   ├── bdd100k_images_100k/
│   │   └── 100k/
│   │       ├── train/  (70,000 images)
│   │       ├── val/    (10,000 images)
│   │       └── test/   (20,000 images)
│   └── bdd100k_labels/
│       └── 100k/
│           ├── train.json
│           ├── val.json
│           └── test.json
└── src/
    └── task2a2/
        └── RT-DETR/
            └── rtdetrv2_pytorch/
```

## 5. Configure RT-DETR for BDD100K

The configuration files are already set up in this repository:

### Dataset Configuration
File: `configs/dataset/bdd_detection.yml`

```yaml
task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: 10
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: ../../../../data/bdd100k_images_100k/100k/train/
    ann_file: ../../../../data/bdd100k_labels/100k/train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: ../../../../data/bdd100k_images_100k/100k/val/
    ann_file: ../../../../data/bdd100k_labels/100k/val.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
```

### Training Configuration
File: `configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml`

Key settings:
- Model: RT-DETRv2-S (ResNet-18 backbone)
- Epochs: 120
- Batch size: Adjust based on your GPU memory
  - 6GB GPU: `total_batch_size: 2` (train), `4` (val)
  - 8GB GPU: `total_batch_size: 4` (train), `8` (val)
  - 11GB+ GPU: `total_batch_size: 8` (train), `16` (val)

## 6. Adjust Batch Size for Your GPU

Edit `configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml`:

```yaml
train_dataloader:
  total_batch_size: 2  # Adjust based on your GPU memory
  # ... rest of config

val_dataloader:
  total_batch_size: 4  # Adjust based on your GPU memory
  # ... rest of config
```

## 7. Start Training

### Single GPU Training

```bash
cd src/task2a2/RT-DETR/rtdetrv2_pytorch

# Basic training
python3 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml --seed=0 --use-amp

# Run in background and save logs
python3 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml --seed=0 --use-amp 2>&1 | tee train.log &
```

### Multi-GPU Training

```bash
cd src/task2a2/RT-DETR/rtdetrv2_pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=9909 \
  --nproc_per_node=4 \
  tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml \
  --use-amp \
  --seed=0 \
  &> train.log 2>&1 &
```

### Training Arguments

- `-c, --config`: Path to config file (required)
- `--use-amp`: Enable automatic mixed precision (recommended)
- `--seed`: Random seed for reproducibility
- `-r, --resume`: Resume from checkpoint path
- `-t, --tuning`: Fine-tune from checkpoint path
- `--output-dir`: Override output directory
- `--test-only`: Run evaluation only (no training)

## 8. Monitor Training

### Check Training Progress

```bash
# View live training logs
tail -f train.log

# Or if running in background
tail -f /path/to/output/file
```

### TensorBoard (if configured)

```bash
tensorboard --logdir=./output/rtdetrv2_r18vd_120e_bdd --port=6006
```

### Expected Training Time

- **Single epoch**: ~50-60 minutes (on RTX 3090, batch_size=2)
- **Full 120 epochs**: ~100-120 hours

### GPU Memory Usage

- RT-DETRv2-S with batch_size=2: ~2-3GB VRAM
- RT-DETRv2-S with batch_size=4: ~4-5GB VRAM

## 9. Checkpoints and Output

Training outputs are saved to: `./output/rtdetrv2_r18vd_120e_bdd/`

**Files:**
- `checkpoint.pth` - Latest checkpoint
- `checkpoint_best.pth` - Best model based on validation mAP
- `model_final.pth` - Final model after all epochs
- TensorBoard event files

## 10. Evaluation

### Evaluate Trained Model

```bash
cd src/task2a2/RT-DETR/rtdetrv2_pytorch

python3 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml \
  -r ./output/rtdetrv2_r18vd_120e_bdd/model_final.pth \
  --test-only
```

## 11. Troubleshooting

### Out of Memory Error

**Solution:** Reduce batch size in config file

```yaml
train_dataloader:
  total_batch_size: 1  # Reduce to 1
```

Or reduce image size:
```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [512, 512]}  # Instead of [640, 640]
```

### CUDA Not Available

Check PyTorch installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

Reinstall PyTorch with correct CUDA version.

### File Not Found Errors

Verify dataset paths in `configs/dataset/bdd_detection.yml` match your actual data location.

### Slow Data Loading

Increase number of workers:
```yaml
train_dataloader:
  num_workers: 8  # Increase based on CPU cores
```

## 12. BDD100K Classes

The model is configured for 10 object classes:

| Class ID | Class Name |
|----------|------------|
| 0 | car |
| 1 | traffic sign |
| 2 | traffic light |
| 3 | person |
| 4 | truck |
| 5 | bus |
| 6 | bike |
| 7 | rider |
| 8 | motor |
| 9 | train |

## 13. Quick Start Script

Create a `setup.sh` script:

```bash
#!/bin/bash

# Install dependencies
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd src/task2a2/RT-DETR/rtdetrv2_pytorch
pip3 install -r requirements.txt

# Verify setup
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Setup complete! Ready to start training."
echo "Run: python3 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_bdd.yml --use-amp --seed=0"
```

Make it executable and run:
```bash
chmod +x setup.sh
./setup.sh
```

## 14. Additional Resources

- **RT-DETR Paper**: https://arxiv.org/abs/2304.08069
- **RT-DETRv2 Paper**: https://arxiv.org/abs/2407.17140
- **BDD100K Dataset**: https://bdd-data.berkeley.edu/
- **Original RT-DETR Repo**: https://github.com/lyuwenyu/RT-DETR

## Support

For issues specific to:
- **BDD dataset**: Check `src/task2/convert_data.py`
- **RT-DETR model**: See RT-DETR's official documentation
- **Training configuration**: Refer to `configs/rtdetrv2/include/` for default settings
