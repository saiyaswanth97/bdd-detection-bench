# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List, Tuple
import numpy as np
import random
import shutil
import os

from src.utils.parse_annotations import parse_annotations, CLASSES


def compute_class_bitmask(files: List[str]) -> Tuple[List[str], List[int]]:
    """
    For each image, compute a bitmask representing the presence of classes.
    1010 -> 1*1+0*2+1*4+0*8
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        Tuple[List[str], List[int]]: A tuple containing:
            - List of sample IDs corresponding to each image.
            - List of integers representing the class bitmask for each image.
    """

    names = []
    out = []

    for file in tqdm(files, desc="Processing annotation files"):
        name, annotations = parse_annotations(str(file))

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

        names.append(name)
        out.append(classes_encoded)

    return names, out


def compute_instance_occurance(files: List[str]) -> Tuple[List[str], dict]:
    """
    For each image, compute number of objects per class.
    Args:
        files (List[str]): List of paths to annotation JSON files.
    Returns:
        Tuple[List[str], dict]: A tuple containing:
            - List of sample IDs corresponding to each image.
            - Dictionary with class names as keys and lists of instance counts as values.
    """

    names = []
    out = {class_name: [] for class_name in CLASSES.keys()}

    for file in tqdm(files, desc="Processing annotation files"):
        name, annotations = parse_annotations(str(file))
        names.append(name)

        image_class_counts = {class_name: 0 for class_name in CLASSES.keys()}

        for annotation in annotations:
            class_name = annotation.class_name
            image_class_counts[class_name] += 1

        for class_name, count in image_class_counts.items():
            out[class_name].append(count)

    return names, out


def plot_presence_pdf(
    class_bitmask: List[int],
    file_name: str,
    start_class: int = 0,
    end_class: int = len(CLASSES) - 1,
) -> None:
    """
    Plots the probability density function of class presence across images using a histogram.
    Args:
        class_bitmask (List[int]): List of integers representing the class bitmask for each image.
        file_name (str): Path to save the generated plot.
        start_class (int): The starting index of classes to include in the plot.
        end_class (int): The ending index of classes to include in the plot.
    """
    x_start = 0
    x_end = 2 ** (end_class + 1)
    bin_width = 2**start_class
    plt.figure(figsize=(10, 6))
    plt.hist(
        class_bitmask,
        bins=np.arange(x_start, x_end + 1, bin_width) - 0.5,
        density=True,
        alpha=0.7,
        color="blue",
    )
    plt.title("Class Distribution per Image")
    plt.xlabel("Encoded Class Presence")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    # plt.xticks(np.arange(x_start, x_end + 1, bin_width))
    plt.savefig(file_name)
    # plt.show()
    plt.close()


def query_presence_sample_id(
    files: List[str], class_bitmask: List[int], query_bitmask: List[int]
) -> List[str]:
    """
    Queries the sample IDs of images that match the specified class presence bitmask.
    Args:
        files (List[str]): List of paths to annotation JSON files.
        class_bitmask (List[int]): List of integers representing the class bitmask for each image.
        query_bitmask (List[int]): List of integers representing the desired class presence bitmask.
            1 -> Should be present. 0 -> Doesn't matter. -1 -> Should be absent.
            Example: [-1, 0, 0, 0, 0, ...] -> Cars are absent, other classes can be present or absent.
            Example: [1, -1, -1, 0, 0, ...] -> Cars should be present, light & sign are absent.
    Returns:
        List[str]: List of sample IDs that match the query bitmask.
    """
    matching_sample_ids = []
    for file, bitmask in zip(files, class_bitmask):
        match = True
        for i in range(len(query_bitmask)):
            if query_bitmask[i] == 1 and (bitmask & (1 << i)) == 0:
                match = False  # Class should be present but is not
                break
            elif query_bitmask[i] == -1 and (bitmask & (1 << i)) != 0:
                match = False  # Class should be absent but is present
                break
        if match:
            sample_id = file
            matching_sample_ids.append(sample_id)
    return matching_sample_ids


def query_frequency_sample_id(
    files: List[str], class_count: dict, query: Tuple[int, int]
) -> List[str]:
    """
    Get the files that have a certain number of instances of a class.
    Args:
        files (List[str]): List of paths to annotation JSON files.
        class_count (dict): Dictionary with class names as keys and lists of instance counts as values.
        query (Tuple[int, int]): Tuple containing the class index and the desired count (class_index, count).
    Returns:
        List[str]: List of sample IDs that match the query.
    """
    sample_ids = []
    class_index, desired_count = query
    class_name = list(CLASSES.keys())[class_index]
    for file, count in zip(files, class_count[class_name]):
        if count == desired_count:
            sample_ids.append(file)
    return sample_ids


def copy_random_file(
    src_folder: str, files: List[str], dst_folder: str, output_name: str
) -> None:
    """
    Copies a random file from the source folder to the destination folder with a specified output name.
    Args:
        src_folder (str): The folder where the original files are located.
        files (List[str]): List of file names to choose from.
        dst_folder (str): The folder where the file should be copied to.
        output_name (str): The name of the copied file in the destination folder.
    Raises:
        ValueError: If the list of files is empty.
        FileNotFoundError: If the source folder or source file does not exist.
    """
    if not files:
        raise ValueError("The list of files is empty. Cannot copy a random file.")
    if not os.path.exists(src_folder):
        raise FileNotFoundError(
            f"Source folder not found: {src_folder}. Please update the path."
        )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    file_to_copy = random.choice(files)
    src_file_path = os.path.join(src_folder, f"{file_to_copy}.jpg")
    if not os.path.exists(src_file_path):
        raise FileNotFoundError(
            f"Source file not found: {src_file_path}. Please check the file name and path."
        )
    dst_file_path = os.path.join(dst_folder, f"{output_name}.jpg")
    shutil.copy(src_file_path, dst_file_path)


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

    image_names, train_class_bitmask = compute_class_bitmask(train_files)

    plot_presence_pdf(
        train_class_bitmask, "src/task1/images/class_presence_distribution.png"
    )
    plot_presence_pdf(
        train_class_bitmask,
        "src/task1/images/class_presence_distribution_no_train.png",
        start_class=0,
        end_class=8,
    )
    plot_presence_pdf(
        train_class_bitmask,
        "src/task1/images/class_presence_distribution_cars_signs_lights.png",
        start_class=0,
        end_class=2,
    )
    plot_presence_pdf(
        train_class_bitmask,
        "src/task1/images/class_presence_distribution_no_cars_signs_lights.png",
        start_class=3,
        end_class=8,
    )

    image_name, train_instance_occurance = compute_instance_occurance(train_files)

    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(CLASSES.keys()):
        plt.subplot(3, 4, i + 1)
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
        plt.yscale("log")
        plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig("src/task1/images/all_class_frequency_distribution.png")
    # plt.show()
    plt.close()

    random.seed(42)
    SRC_IMAGES_FOLDER = "data/bdd100k_images_100k/100k/train"
    DST_IMAGES_FOLDER = "src/task1/images"

    presence_queries = [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "sample_train"),
        ([-1, -1, -1, 0, 0, 0, 0, 0, 0, 0], "sample_no_car_sign_light"),
    ]
    for query_bitmask, output_name in presence_queries:
        copy_random_file(
            SRC_IMAGES_FOLDER,
            query_presence_sample_id(image_names, train_class_bitmask, query_bitmask),
            DST_IMAGES_FOLDER,
            output_name,
        )

    frequency_queries = [
        ((0, 40), "image_40_cars"),  # 40 cars
        ((1, 30), "image_30_signs"),  # 30 signs
        ((3, 40), "image_40_persons"),  # 40 persons
        ((6, 15), "image_15_bikes"),  # 15 bikes
        ((8, 10), "image_10_motorcycles"),  # 10 motorcycles
        ((9, 4), "image_4_trains"),  # 4 trains
    ]
    for query, output_name in frequency_queries:
        copy_random_file(
            SRC_IMAGES_FOLDER,
            query_frequency_sample_id(
                image_names, train_instance_occurance, query=query
            ),
            DST_IMAGES_FOLDER,
            output_name,
        )
