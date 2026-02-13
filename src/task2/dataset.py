# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Dataset registration and loading for BDD100K dataset in Detectron2 format."""

import os
from typing import List, Dict

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, DatasetMapper, transforms


def make_coco_dicts(mode: str) -> List[Dict]:
    """
    Convert COCO annotations to Detectron2 format.
    Args:
        mode (str): One of "train", "val", or "test" to specify the dataset split.
    Returns:
        List[Dict]: List of dictionaries in Detectron2 format.
    Raises:
        ValueError: If the mode is not one of "train", "val", or "test".
        FileNotFoundError: If the annotation file or image directory does not exist.
    """
    DATA_FOLDER = "data/"
    DATASETS = {
        "train": {
            "json": "bdd100k_labels/100k/train.json",
            "img_dir": "bdd100k_images_100k/100k/train/",
            "name": "bdd_dataset_train",
        },
        "val": {
            "json": "bdd100k_labels/100k/val.json",
            "img_dir": "bdd100k_images_100k/100k/val/",
            "name": "bdd_dataset_val",
        },
        "test": {
            "json": "bdd100k_labels/100k/test.json",
            "img_dir": "bdd100k_images_100k/100k/test/",
            "name": "bdd_dataset_test",
        },
    }

    if mode not in DATASETS:
        raise ValueError(
            f"Invalid mode {mode}. Expected one of {list(DATASETS.keys())}."
        )

    json_file = DATA_FOLDER + DATASETS[mode]["json"]
    img_dir = DATA_FOLDER + DATASETS[mode]["img_dir"]
    dataset_name = DATASETS[mode]["name"]

    if dataset_name in DatasetCatalog.list():
        print(f"Dataset {dataset_name} already registered. Returning existing dataset.")
        return DatasetCatalog.get(dataset_name)
    else:
        print(
            f"Registering dataset {dataset_name} with annotations from {json_file} and images from {img_dir}."
        )
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Annotation file {json_file} does not exist.")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory {img_dir} does not exist.")
        register_coco_instances(
            dataset_name,
            {},
            json_file,
            img_dir,
        )
        print(f"Dataset {dataset_name} registered successfully.")
        return DatasetCatalog.get(dataset_name)


def get_transform(is_train: bool) -> DatasetMapper:
    """
    Get data augmentation transforms for training or validation. Original img size: 1280x720.
    Args:
        is_train (bool): Whether to return transforms for training or validation.
    Returns:
        DatasetMapper: A Detectron2 DatasetMapper with the specified augmentations.
    """
    if is_train:
        augmentation = [
            transforms.ResizeShortestEdge(
                short_edge_length=(480, 720), max_size=1280, sample_style="range"
            ),
            transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            transforms.RandomBrightness(0.8, 1.2),
            transforms.RandomContrast(0.8, 1.2),
            transforms.RandomSaturation(0.8, 1.2),
            transforms.RandomLighting(0.7),
        ]
        return DatasetMapper(
            is_train=True,
            augmentations=augmentation,
            image_format="BGR",
            use_instance_mask=False,
            use_keypoint=False,
        )
    else:
        augmentation = [
            transforms.ResizeShortestEdge(short_edge_length=720, max_size=1280),
        ]
        return DatasetMapper(
            is_train=False,
            augmentations=augmentation,
            image_format="BGR",
            use_instance_mask=False,
            use_keypoint=False,
        )


if __name__ == "__main__":
    import random
    import cv2
    from matplotlib import pyplot as plt

    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    dataset_dicts = make_coco_dicts("train")
    print(f"Number of samples in dataset: {len(dataset_dicts)}")
    print(f"Sample entry: {dataset_dicts[0]}")

    # Visualize a few samples
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        print(f"Visualizing {d['file_name']}")
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get("bdd_dataset_train"),
            scale=1.0,
        )
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10, 6))
        plt.imshow(out.get_image())
        plt.axis("off")
        plt.show()

    meta = MetadataCatalog.get("bdd_dataset_train")
    print("Thing classes:", meta.thing_classes)
    print(
        "Dataset ID to contiguous ID mapping:", meta.thing_dataset_id_to_contiguous_id
    )

    from detectron2.data import build_detection_train_loader
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("bdd_dataset_train",)
    cfg.DATALOADER.NUM_WORKERS = 2

    train_loader = build_detection_train_loader(
        cfg,
        mapper=get_transform(is_train=True),
    )
    print("\nTrain loader created successfully!")
    print("Testing data loading by fetching one batch...")
    batch = next(iter(train_loader))

    print("\n=== Visualizing augmented images with ground truth ===")
    for i in range(min(len(batch), 2)):  # Visualize up to 2 samples
        # Get image tensor (C, H, W) and convert to numpy (H, W, C)
        img_tensor = batch[i]["image"]
        img = img_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")

        print(
            f"Image shape after augmentation: {img.shape}, Number of instances: {len(batch[i]['instances'])}"
        )

        # Create visualizer - img is in RGB format from DatasetMapper
        visualizer = Visualizer(
            img,
            metadata=MetadataCatalog.get("bdd_dataset_train"),
            scale=1.0,
        )

        # Draw ground truth instances
        instances = batch[i]["instances"].to("cpu")
        boxes = (
            instances.gt_boxes
            if hasattr(instances, "gt_boxes")
            else instances.pred_boxes
        )
        labels = (
            instances.gt_classes
            if hasattr(instances, "gt_classes")
            else instances.pred_classes
        )
        out = visualizer.overlay_instances(boxes=boxes, labels=labels)

        plt.figure(figsize=(12, 7))
        plt.imshow(out.get_image())
        plt.axis("off")
        plt.title(f"Augmented Image {i + 1} with Ground Truth")
        plt.tight_layout()
        plt.show()
