# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

# import cv2

# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.utils.visualizer import Visualizer


# register_coco_instances(
#     "my_dataset", {},
#     "data/bdd100k_labels/100k/train.json",
#     "data/bdd100k_images_100k/100k/train/")

# print("Dataset registered successfully!")


# dataset_dicts = DatasetCatalog.get("my_dataset")

# print(len(dataset_dicts))
# print(dataset_dicts[0])

# # for d in dataset_dicts[:3]:
# #     img = cv2.imread(d["file_name"])
# #     print(d["file_name"])
# #     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("my_dataset"), scale=0.5)
# #     out = visualizer.draw_dataset_dict(d)
# #     cv2.imshow('Sample Image', out.get_image()[:, :, ::-1])
# #     cv2.waitKey(0)

# meta = MetadataCatalog.get("my_dataset")
# print(meta.thing_classes)
# print(meta.thing_dataset_id_to_contiguous_id)


import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# ---- Config ----
cfg = get_cfg()

# Load a model config from model zoo
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)

# Use COCO pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

# ---- Load dataset ----
register_coco_instances(
    "my_dataset",
    {},
    "data/bdd100k_labels/100k/train.json",
    "data/bdd100k_images_100k/100k/train/",
)

# Load the dataset to populate metadata
_ = DatasetCatalog.get("my_dataset")
print("Thing classes:", MetadataCatalog.get("my_dataset").thing_classes)

# ---- Dataset ----
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ("my_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2

# ---- Training params ----
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []  # no LR decay

# ---- Model specifics ----
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("my_dataset").thing_classes)

# ---- Output ----
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ---- Train ----
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
