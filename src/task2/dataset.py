# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Dataset registration and loading for BDD100K dataset in Detectron2 format."""

import os
from typing import List, Dict

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog


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
