# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List
import numpy as np

from src.utils.parse_annotations import parse_annotations, CLASSES


def compute_bbox_area_distribution(files: List[str]) -> dict:
    """
    Calculate the distribution of bounding box sizes from a list of annotation files.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        dict: Dictionary mapping class names to lists of bounding box areas.
    """
    bbox_distribution = {class_name: [] for class_name in CLASSES.keys()}

    for file in tqdm(files, desc="Processing annotation files for bbox distribution"):
        name, annotations = parse_annotations(str(file))
        for annotation in annotations:
            class_name = annotation.class_name
            bbox_area = annotation.bbox.area / (
                1280 * 720
            )  # Normalize area by image size
            bbox_distribution[class_name].append(bbox_area)

    return bbox_distribution


def compute_bbox_region_distribution(files: List[str]) -> dict:
    """
    Calculate the distribution of bounding box regions from a list of annotation files.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        dict: Dictionary mapping class names to 2D arrays representing the distribution of bounding box regions.
    """
    images = {
        class_name: np.zeros((720, 1280), dtype=np.uint32)
        for class_name in CLASSES.keys()
    }

    for file in tqdm(files, desc="Processing annotation files for bbox distribution"):
        name, annotations = parse_annotations(str(file))
        for annotation in annotations:
            class_name = annotation.class_name
            x1, y1, x2, y2 = (
                int(annotation.bbox.x1),
                int(annotation.bbox.y1),
                int(annotation.bbox.x2),
                int(annotation.bbox.y2),
            )
            images[class_name][y1:y2, x1:x2] += 1

    for class_name in CLASSES.keys():
        images[class_name] = (
            images[class_name].astype(np.float64) / 16 / len(files) * 70000
        )
        images[class_name] = np.clip(images[class_name], 0, 255).astype(np.uint8)

    return images


def compute_bbox_aspect_distribution(files: List[str]) -> dict:
    """
    Calculate the distribution of bounding box aspect ratios from a list of annotation files.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        dict: Dictionary mapping class names to lists of bounding box aspect ratios.
    """
    aspect_distribution = {class_name: [] for class_name in CLASSES.keys()}

    for file in tqdm(
        files, desc="Processing annotation files for bbox aspect distribution"
    ):
        name, annotations = parse_annotations(str(file))
        for annotation in annotations:
            class_name = annotation.class_name
            width = annotation.bbox.x2 - annotation.bbox.x1
            height = annotation.bbox.y2 - annotation.bbox.y1
            if height > 0:
                aspect_ratio = width / height
                aspect_distribution[class_name].append(aspect_ratio)

    return aspect_distribution


if __name__ == "__main__":
    from pathlib import Path

    train_data_folder = "data/bdd100k_labels/100k/train"
    val_data_folder = "data/bdd100k_labels/100k/val"
    if not Path(train_data_folder).exists():
        raise FileNotFoundError(
            f"Train data folder not found: {train_data_folder}. Please update the path."
        )
    if not Path(val_data_folder).exists():
        raise FileNotFoundError(
            f"Validation data folder not found: {val_data_folder}. Please update the path."
        )

    train_files = list(Path(train_data_folder).glob("*.json"))
    if not train_files:
        raise FileNotFoundError(
            f"No JSON files found in the train data folder: {train_data_folder}. Please check the contents."
        )
    val_files = list(Path(val_data_folder).glob("*.json"))
    if not val_files:
        raise FileNotFoundError(
            f"No JSON files found in the validation data folder: {val_data_folder}. Please check the contents."
        )

    iterations = [[train_files, "Train"], [val_files, "Validation"]]

    for files, split in iterations:
        class_bbox_distribution = compute_bbox_area_distribution(files)

        fig = plt.figure(figsize=(12, 8))
        for i, (class_name, areas) in enumerate(class_bbox_distribution.items()):
            plt.subplot(4, 3, i + 1)
            weights = np.ones(len(areas)) / len(areas)
            plt.hist(areas, bins=50, color="blue", alpha=0.7, weights=weights)
            plt.title(f"{class_name} Area Distribution")
            plt.xlabel("Area")
            plt.ylabel("Frequency %")
            plt.yscale("log")
            plt.ylim(1e-6, 1)
            plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"src/task1/images/bbox_area_distribution_{split.lower()}.png")

        traffic_iterations = [
            ["traffic light", "traffic_light"],
            ["traffic sign", "traffic_sign"],
        ]
        if split == "Train":
            for class_name, file_name in traffic_iterations:
                bbox_distribution = class_bbox_distribution[class_name]
                plt.figure(figsize=(8, 6))
                weights = np.ones(len(bbox_distribution)) / len(bbox_distribution)
                plt.hist(
                    bbox_distribution,
                    bins=np.arange(0, 0.06, 0.0005),
                    color="blue",
                    alpha=0.7,
                    weights=weights,
                )
                plt.title(f"{class_name.capitalize()} Area Distribution")
                plt.xlabel("Area")
                plt.ylabel("Frequency %")
                plt.yscale("log")
                plt.ylim(1e-6, 1)
                plt.grid(axis="y", alpha=0.75)
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"src/task1/images/bbox_area_distribution_{file_name}.png")

        class_bbox_region_distribution = compute_bbox_region_distribution(files)

        plt.figure(figsize=(10, 8))
        for i, (class_name, image) in enumerate(class_bbox_region_distribution.items()):
            plt.subplot(4, 3, i + 1)
            plt.imshow(image, cmap="gray")
            plt.title(f"{class_name} Region Distribution")
            plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"src/task1/images/bbox_region_distribution_{split.lower()}.png")

        class_bbox_aspect_distribution = compute_bbox_aspect_distribution(files)
        fig = plt.figure(figsize=(12, 8))
        plt.boxplot(
            [
                aspect_ratios
                for aspect_ratios in class_bbox_aspect_distribution.values()
            ],
            labels=class_bbox_aspect_distribution.keys(),
            showfliers=False,
        )
        plt.title(f"Bounding Box Aspect Ratio Distribution ({split})")
        plt.xlabel("Class")
        plt.ylabel("Aspect Ratio (Width/Height)")
        plt.savefig(f"src/task1/images/bbox_aspect_distribution_{split.lower()}.png")
