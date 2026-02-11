# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List
import numpy as np

from src.utils.parse_annotations import parse_annotations, CLASSES


def compute_class_distribution(files: List[str], all_occurrences: bool = False) -> dict:
    """
    Calculate the class distribution from a list of annotation files.
    Args:
        files (List[str]): List of paths to annotation JSON files.
        all_occurrences (bool): Whether to count all occurrences of each class or just once per image.
    Returns:
        dict: Dictionary mapping class names to their occurrence counts.
    """

    class_counts = {class_name: 0 for class_name in CLASSES.keys()}

    for file in tqdm(files, desc="Processing annotation files"):
        name, annotations = parse_annotations(str(file))

        if not all_occurrences:
            seen_classes_in_image = (
                set()
            )  # To track classes that have already been counted for this image
            for annotation in annotations:
                class_name = annotation.class_name
                if class_name not in seen_classes_in_image:
                    class_counts[class_name] += 1
                    seen_classes_in_image.add(class_name)
        else:
            for annotation in annotations:
                class_name = annotation.class_name
                class_counts[class_name] += 1

    return class_counts


def compute_class_cooccurrence_matrix(files: List[str]) -> np.ndarray:
    """
    Calculate the co-occurrence of classes from a list of annotation files.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        np.ndarray: A numpy array representing the co-occurrence matrix of classes.
    """

    cooccurrence_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)

    for file in tqdm(files, desc="Processing annotation files for co-occurrence"):
        name, annotations = parse_annotations(str(file))
        image_classes = set(annotation.class_name for annotation in annotations)
        for class1 in image_classes:
            for class2 in image_classes:
                cooccurrence_matrix[CLASSES[class1]][CLASSES[class2]] += 1

    # Apply log with handling for zero values (add small epsilon to avoid log(0))
    cooccurrence_matrix = np.where(
        cooccurrence_matrix > 0, np.log(cooccurrence_matrix + 1), 0
    )

    return cooccurrence_matrix


def plot_class_distribution(
    class_distribution: dict,
    title: str,
    filename: str,
    log_scale: bool,
    show_plot: bool = False,
) -> None:
    """
    Plot the class distribution as a bar chart.
    Args:
        class_distribution (dict): Dictionary with class names as keys and counts as values.
        title (str): Title of the plot.
        filename (str): Path to save the plot image.
        log_scale (bool): Whether to use logarithmic scale for the y-axis.
        show_plot (bool): Whether to display the plot after saving.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution.keys(), class_distribution.values(), color="blue")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)
    if log_scale:
        plt.yscale(
            "log"
        )  # Use logarithmic scale for better visibility of smaller classes
    plt.tight_layout()
    plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()


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

    train_counts_per_image = compute_class_distribution(
        train_files, all_occurrences=False
    )
    train_counts_all = compute_class_distribution(train_files, all_occurrences=True)
    train_avg_per_image = {
        class_name: (train_counts_all[class_name] / train_counts_per_image[class_name])
        if train_counts_per_image[class_name] > 0
        else 0
        for class_name in CLASSES.keys()
    }

    val_files = list(Path(val_data_folder).glob("*.json"))
    if not val_files:
        raise FileNotFoundError(
            f"No JSON files found in the validation data folder: {val_data_folder}. Please check the contents."
        )

    val_counts_per_image = compute_class_distribution(val_files, all_occurrences=False)
    val_counts_all = compute_class_distribution(val_files, all_occurrences=True)
    train_val_ratio_all = {
        class_name: (train_counts_all[class_name] / val_counts_all[class_name])
        * (len(val_files) / len(train_files))
        if val_counts_all[class_name] > 0
        else 0
        for class_name in CLASSES.keys()
    }
    train_val_ratio_per_image = {
        class_name: (
            train_counts_per_image[class_name]
            / val_counts_per_image[class_name]
            * (len(val_files) / len(train_files))
        )
        if val_counts_per_image[class_name] > 0
        else 0
        for class_name in CLASSES.keys()
    }

    # Plotting the class distribution
    plot_class_distribution(
        train_counts_per_image,
        "Class Distribution (Once per Image) in BDD100K Train Set",
        "src/task1/images/class_cumulative_frequency.png",
        log_scale=True,
    )
    plot_class_distribution(
        train_counts_all,
        "Class Distribution (All Occurrences) in BDD100K Train Set",
        "src/task1/images/class_cumulative_presence.png",
        log_scale=True,
    )
    plot_class_distribution(
        train_avg_per_image,
        "Average Occurrence of Each Class per Image in BDD100K Train Set",
        "src/task1/images/class_occurrence_per_image.png",
        log_scale=False,
    )
    plot_class_distribution(
        train_val_ratio_all,
        "Ratio of Occurrence of Each Class in Validation Set to Train Set (All Occurrences)",
        "src/task1/images/class_occurrence_ratio_all.png",
        log_scale=False,
    )
    plot_class_distribution(
        train_val_ratio_per_image,
        "Ratio of Occurrence of Each Class in Validation Set to Train Set (Once per Image)",
        "src/task1/images/class_occurrence_ratio_per_image.png",
        log_scale=False,
    )

    class_cooccurrence_matrix = compute_class_cooccurrence_matrix(train_files)
    plt.figure(figsize=(12, 10))
    plt.imshow(class_cooccurrence_matrix, cmap="viridis")
    plt.colorbar(label="Co-occurrence Rate")
    plt.xticks(ticks=np.arange(len(CLASSES)), labels=CLASSES.keys(), rotation=45)
    plt.yticks(ticks=np.arange(len(CLASSES)), labels=CLASSES.keys())
    plt.title("Co-occurrence of Classes in BDD100K Test Set")
    plt.tight_layout()
    plt.savefig("src/task1/images/class_cooccurrence.png")
