# Model Selection

## Requirements
- The data is heavily imbalanced
- Multiple objects in image (~15)
- Small objects
  - Traffic light smaller than 32x32, in full image resolution >50%
  - These objects are quite important for ADAS
- Real-time & edge deployment

**For this assignment**
- Pretrained model
- Ease to train
- SOTA

### Architecture
- CNN vs Transformers
  - Transformers can achieve higher accuracy for same data + scaling law
  - But transformers are much slower than CNN. Not fully hardware+CUDA optimized.
  - Best for realtime - mix of CNN & transformer. CNN for early layer & transformers for later

### Option 1 - YOLO:
- TLDR: No
- Pros:
  - Extremely fast
  - Multiple pretrained models
  - SOTA for COCO
- Cons:
  - Multiple object with heavy overlap. As YOLO relies on NMS it would perform bad for our case
  - Multiple small objects. YOLO is trained on downsized version of original. The final layer in pyramidal will at most have one pixel feature for the object. So it would perform bad for this case.

### Option 2 - RT-DETR:
- TLDR: Trained and results are in `src/task3/detr_train`
- Pros:
  - Extremely fast
  - Code + model available by authors
  - SOTA for COCO
  - Does the hybrid CNN+transformer for backbone. Tranformer based detection head is lot cleaner (No anchors).
- Cons:
  - Need a bulky GPU to train

### Option 3 - faster_rcnn_R_50_FPN+Detectron2:
- TLDR: Trained for task3 results (`src/task3/r_cnn_train`). Code: `src/task2`.
- Pros:
  - Fast & ease to train
  - Ease to develop using detectron2 framework
- Cons:
  - Worse performance
  - Not edge deployable
- Reason to choose:
  - Detectron2 based training is highly optimized to train 1 epoch (70K) took 25 mins. I can iterate faster and generate visualizations.

### Option 4 - Torch based faster_rcnn_R_50_FPN:
- Code: `src/task2_bonus`
- Implement basic training using torch + torch vision
- Custom metrics & dataloader
- 10x slower on laptop; 2x slower on RTX5090

---

# Detectron2 - faster_rcnn_R_50_FPN:



## Framework Selection

- I will use Torch/Torch Lightning based frameworks
- Popular - Detectron2 & mmdetection
- Chose mmdetection intially, the build issue are bad & the repo is no longer maintained.
- Installation: `src/task2_bonus/docs/Detectron2_install_guide.md`

---

## How to Run

### Prerequisites
1. Install Detectron2 (see installation guide: `src/task2_bonus/docs/Detectron2_install_guide.md`)
2. Ensure BDD100K dataset is downloaded and available in the `data/` directory

### Dataset Preparation
Convert BDD100K annotations to COCO format:
```bash
python3 -m src.task2.tools.convert_data
```

This script will:
- Convert annotations from BDD100K format to COCO format
- Process train, val, and test splits
- Generate `train.json`, `val.json`, and `test.json` in `data/bdd100k_labels/100k/`
- Each JSON file contains images, annotations, and category mappings in COCO format

**Note:** This conversion only needs to be run once before training.

### Training
```bash
# Train Faster R-CNN with Detectron2
python3 -m src.task2.main
```

**Configuration:**
- Default epochs: 10
- Batch size: 24
- Base learning rate: Auto-scaled based on batch size
- Output directory: `./output`
- Checkpoints saved every half epoch
- Validation runs every epoch

### Evaluation
```bash
# Evaluate trained models
python3 -m src.task2.eval_models
```

### Visualization
```bash
# Generate visualizations
python3 -m src.task2.gen_viz

# Plot evaluation results
python3 -m src.task2.plot_eval
```

### Key Files
- `main.py`: Training script with custom BDDTrainer
- `dataset.py`: Dataset loading and COCO format conversion
- `eval_models.py`: Model evaluation on validation set
- `visualization.py`: Visualization hooks and utilities
- `gen_viz.py`: Generate prediction visualizations
- `plot_eval.py`: Plot evaluation metrics


### TODO
- [ ] Refactor for image visualization gif
- [ ] Reafactor eval utils
