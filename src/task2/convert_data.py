# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Convert from Bdd format to COCO format"""

import os
import json
from tqdm import tqdm
from src.utils.parse_annotations import CLASSES, parse_annotations


if __name__ == "__main__":
    ANN_FOLDER = "data/bdd100k_labels/100k/"

    FOLDERS = [["train", "train.json"], ["test", "test.json"], ["val", "val.json"]]

    for folder, output in FOLDERS:
        files = os.listdir(ANN_FOLDER + folder)
        output_file = ANN_FOLDER + output
        output_json = {"images": [], "annotations": [], "categories": []}
        ann_id = 0
        image_id = 0

        for file in tqdm(files):
            file_path = ANN_FOLDER + folder + "/" + file
            image_name, labels = parse_annotations(file_path)

            output_json["images"].append(
                {
                    "id": image_id,
                    "file_name": image_name + ".jpg",
                    "width": 1280,
                    "height": 720,
                }
            )

            for label in labels:
                output_json["annotations"].append(
                    {
                        "image_id": image_id,
                        "category_id": label.class_id,
                        "bbox": [
                            label.bbox.x1,
                            label.bbox.y1,
                            label.bbox.x2 - label.bbox.x1,
                            label.bbox.y2 - label.bbox.y1,
                        ],
                        "id": ann_id,
                    }
                )
                ann_id += 1
            image_id += 1

        for class_name, class_id in CLASSES.items():
            output_json["categories"].append({"id": class_id, "name": class_name})

        with open(output_file, "w") as f:
            json.dump(output_json, f)
