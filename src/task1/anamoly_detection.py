# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List
import numpy as np

from src.utils.parse_annotations import parse_annotations, CLASSES


def compute_class_bitmask(files: List[str]) -> List[int]:
    """
    For each image, compute a bitmask representing the presence of classes.
    1010 -> 1*1+0*2+1*4+0*8
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        List[int]: List of integers representing the class bitmask for each image.
    """

    out = []

    for file in tqdm(files, desc="Processing annotation files"):
        _, annotations = parse_annotations(str(file))

        seen_classes_in_image = (
            set()
        )  # To track classes that have already been counted for this image
        for annotation in annotations:
            class_name = annotation.class_name
            if class_name not in seen_classes_in_image:
                seen_classes_in_image.add(class_name)

        classes_encoded = 0
        for i, class_name in enumerate(CLASSES.keys()):
            if class_name in seen_classes_in_image:
                classes_encoded += 2**i

        out.append(classes_encoded)

    return out


def compute_instance_occurance(files: List[str]) -> List[int]:
    """
    For each image, compute number of objects per class.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        List[int]: List of integers representing the class bitmask for each image.
    """

    out = {class_name: [] for class_name in CLASSES.keys()}

    for file in tqdm(files, desc="Processing annotation files"):
        _, annotations = parse_annotations(str(file))

        image_class_counts = {class_name: 0 for class_name in CLASSES.keys()}

        for annotation in annotations:
            class_name = annotation.class_name
            image_class_counts[class_name] += 1

        for class_name, count in image_class_counts.items():
            out[class_name].append(count)

    return out


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

    train_class_bitmask = compute_class_bitmask(train_files)

    plt.figure(figsize=(10, 6))
    plt.hist(
        train_class_bitmask,
        bins=np.arange(0, 2 ** len(CLASSES) + 1) - 0.5,
        density=True,
        alpha=0.7,
        color="blue",
    )
    plt.title("Class Distribution per Image (Train Set)")
    plt.xlabel("Encoded Class Presence")
    plt.ylabel("Density")
    plt.xticks(np.arange(0, 2 ** len(CLASSES), 2 ** (len(CLASSES) - 1)))
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig("train_class_distribution_per_image.png")
    plt.show()
    plt.close()

    train_instance_occurance = compute_instance_occurance(train_files)

    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(CLASSES.keys()):
        plt.subplot(4, 4, i + 1)
        plt.hist(
            train_instance_occurance[class_name],
            bins=np.arange(0, max(train_instance_occurance[class_name]) + 1) - 0.5,
            density=True,
            alpha=0.7,
            color="blue",
        )
        plt.title(f"{class_name} Count per Image")
        plt.xlabel("Count")
        plt.ylabel("Density")
        plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig("train_instance_occurance_per_image.png")
    plt.show()
    plt.close()
