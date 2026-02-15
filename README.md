# bdd-detection-bench

BDD100K dataset analysis and detection benchmark project.

<!-- ## 🔧 Setup

### Install Dependencies

```bash
pip3 install tqdm matplotlib numpy pre-commit
```

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. The hooks include:

- **ruff**: Python linter and code formatter
- **insert-license**: Automatically adds license headers to Python files
- **pydoclint**: Validates Google-style docstrings

#### Install Pre-commit Hooks

```bash
# Install pre-commit (if not already installed)
pip3 install pre-commit

# Install the git hooks
pre-commit install

# Run hooks on all files (dry run)
pre-commit run --all-files
```

The hooks will automatically run on every `git commit`. If any hook fails, the commit will be blocked until issues are resolved. -->

## 📁 Project Structure

```
bdd-detection-bench/
├── src/
│   ├── task1/               # Class distribution analysis
│   ├── task2/               # Object detection with Detectron2
│   ├── task2_bonus/         # Object detection with PyTorch
│   ├── task3/               # Advanced detection experiments
│   └── utils/               # Shared utilities
│       └── parse_annotations.py
├── data/
│   ├── bdd100k_labels/      # Dataset annotations
│   └── bdd100k_images_100k/ # Dataset images
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .license_header          # License template
└── pyproject.toml           # Project configuration
```

## 📊 Tasks

### Task 1: Class Distribution Analysis

Comprehensive analysis of class distribution in the BDD100K dataset, including:
- Class occurrence patterns and distributions
- Train/validation split ratios
- Class co-occurrence matrices
- Anomaly detection and sample extraction

**[→ See Task 1 Documentation](src/task1/README.md)**

**[→ Docker Setup for Task 1](docker/README.md)**

---

### Task 2: Object Detection with Detectron2

Faster R-CNN implementation using Facebook's Detectron2 framework:
- Model selection and architecture comparison (YOLO, RT-DETR, Faster R-CNN)
- Detectron2-based training pipeline
- Dataset conversion (BDD100K → COCO format)
- Custom visualization hooks
- Evaluation and metrics

**[→ See Task 2 Documentation](src/task2/README.md)**

---

### Task 2 Bonus: PyTorch Faster R-CNN

Pure PyTorch implementation with custom components:
- Custom BDDDataset with data augmentation
- Custom metrics (AP/mAP calculation)
- Mixed precision training (AMP)
- TensorBoard logging
- From-scratch training loop

**[→ See Task 2 Bonus Documentation](src/task2_bonus/README.md)**

---

### Task 3: Model Training & Evaluation

Comprehensive training experiments and performance analysis:

#### RT-DETR Training
- RT-DETRv2-S model on BDD100K
- Final mAP: 32.0
- Training loss analysis and hyperparameter tuning
- Small object detection challenges

**[→ See RT-DETR Documentation](src/task3/detr_train/README.md)**

#### Faster R-CNN Training
- Detectron2-based training pipeline
- Final mAP: 29.0
- Per-class and per-size performance analysis
- Qualitative results with GIF visualizations

**[→ See Faster R-CNN Documentation](src/task3/r_cnn_train/README.md)**