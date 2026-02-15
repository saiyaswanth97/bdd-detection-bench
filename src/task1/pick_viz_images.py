# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import random
from pathlib import Path
import shutil

from src.task1.anamoly_detection import (
    query_presence_sample_id,
    query_frequency_sample_id,
    compute_class_bitmask,
    compute_instance_occurance,
)


if __name__ == "__main__":
    data_folder = "data/bdd100k_labels/100k/val"
    if not Path(data_folder).exists():
        raise FileNotFoundError(
            f"Validation data folder not found: {data_folder}. Please update the path."
        )

    files = list(Path(data_folder).glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in the train data folder: {data_folder}. Please check the contents."
        )

    image_names, class_bitmask = compute_class_bitmask(files)
    image_name, instance_occurance = compute_instance_occurance(files)

    random.seed(42)
    SRC_IMAGES_FOLDER = "data/bdd100k_images_100k/100k/val"
    DST_IMAGES_FOLDER = "src/task3/viz_src_images"
    JSON_FOLDER = "data/bdd100k_labels/100k/val"
    DST_JSON_FOLDER = "src/task3/viz_src_jsons"

    presence_queries = [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "sample_train"),
        ([-1, -1, -1, 0, 0, 0, 0, 0, 0, 0], "sample_no_car_sign_light"),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "sample_car"),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "sample_sign"),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], "sample_light"),
        ([1, 1, 1, -1, -1, -1, -1, -1, -1, -1], "sample_car_sign_light"),
        ([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], "sample_motorcycle"),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "sample_bike"),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "sample_person"),
        ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "sample_rider"),
    ]
    for query_bitmask, output_name in presence_queries:
        samples = query_presence_sample_id(image_names, class_bitmask, query_bitmask)
        random_sample = random.choice(samples)

        shutil.copyfile(
            Path(SRC_IMAGES_FOLDER) / (random_sample + ".jpg"),
            Path(DST_IMAGES_FOLDER) / (output_name + ".jpg"),
        )
        shutil.copyfile(
            Path(JSON_FOLDER) / (random_sample + ".json"),
            Path(DST_JSON_FOLDER) / (output_name + ".json"),
        )

    frequency_queries = [
        ((0, 20), "image_20_cars"),  # 20 cars
        ((1, 10), "image_10_signs"),  # 10 signs
        ((2, 10), "image_10_lights"),  # 10 traffic lights
        ((3, 10), "image_10_persons"),  # 10 persons
        ((6, 5), "image_5_bikes"),  # 5 bikes
    ]
    for query, output_name in frequency_queries:
        samples = query_frequency_sample_id(
            image_names, instance_occurance, query=query
        )
        sample = random.choice(samples)

        shutil.copyfile(
            Path(SRC_IMAGES_FOLDER) / (sample + ".jpg"),
            Path(DST_IMAGES_FOLDER) / (output_name + ".jpg"),
        )
        shutil.copyfile(
            Path(JSON_FOLDER) / (sample + ".json"),
            Path(DST_JSON_FOLDER) / (output_name + ".json"),
        )
