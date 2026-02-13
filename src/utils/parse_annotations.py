# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""
Utils for parsing JSON annotations from BDD100K dataset.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Tuple
import time


CLASSES = {
    "car": 0,
    "traffic sign": 1,
    "traffic light": 2,
    "person": 3,
    "truck": 4,
    "bus": 5,
    "bike": 6,
    "rider": 7,
    "motor": 8,
    "train": 9,
}


@dataclass
class BBox:
    """Bounding box in x1, y1, x2, y2 format"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_coco(self) -> List[float]:
        """
        Convert to COCO format [x, y, width, height]
        Returns:
            List[float]: [x, y, width, height]
        """
        return [self.x1, self.y1, self.width, self.height]

    def to_yolo(self, img_width: int, img_height: int) -> List[float]:
        """
        Convert to YOLO format [x_center, y_center, width, height] normalized
        Args:
            img_width (int): Width of the image
            img_height (int): Height of the image
        Returns:
            List[float]: [x_center, y_center, width, height] normalized to [0, 1]
        """
        cx, cy = self.center
        return [
            cx / img_width,
            cy / img_height,
            self.width / img_width,
            self.height / img_height,
        ]


@dataclass
class Annotation:
    """Annotation for a single object in an image"""

    class_name: str
    class_id: int
    bbox: BBox
    occlusion: bool
    truncation: bool
    signal_color: str = None  # Only for traffic lights


def parse_annotations(json_file: str) -> Tuple[str, List[dict]]:
    """
    Parse JSON annotations and return a list of dictionaries with class and bounding box info.
    Args:
        json_file (str): Path to the JSON annotation file
    Returns:
        Tuple[str, List[dict]]: Image name and list of annotations
    Raises:
        FileNotFoundError: If the annotation file does not exist.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Annotation file {json_file} does not exist.")

    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = []
    name = data["name"]
    for item in data["frames"][0]["objects"]:
        category_name = item["category"]
        if category_name not in CLASSES:
            # TODO I am skipping all other classes. Cleaner would be to make a list of allowed and unallowed classes.
            continue  # Skip unknown categories
        bbox = BBox(
            x1=item["box2d"]["x1"],
            y1=item["box2d"]["y1"],
            x2=item["box2d"]["x2"],
            y2=item["box2d"]["y2"],
        )
        class_id = CLASSES[category_name]
        occlusion = item["attributes"]["occluded"]
        truncation = item["attributes"]["truncated"]
        signal_color = item["attributes"]["trafficLightColor"]
        annotations.append(
            Annotation(
                class_name=category_name,
                class_id=class_id,
                bbox=bbox,
                occlusion=occlusion,
                truncation=truncation,
                signal_color=signal_color if category_name == "traffic light" else None,
            )
        )

    return (name, annotations)


if __name__ == "__main__":
    from pathlib import Path

    data_folder = "data/"
    file = "bdd100k_labels/100k/test/fda06ff3-d7816e0d.json"

    test_file = Path(data_folder) / file
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. Please update the path."
        )

    start = time.time()
    name, annotations = parse_annotations(str(test_file))
    end = time.time()
    print(f"Parsing took {end - start:.4f} seconds")
    print(f"Image Name: {name}, Number of Annotations: {len(annotations)}")
    for ann in annotations:
        if ann.class_name == "traffic light":
            print(
                f"Class: {ann.class_name}, ID: {ann.class_id}, BBox: {ann.bbox.to_coco()}, "
                f"Occlusion: {ann.occlusion}, Truncation: {ann.truncation}, "
                f"Signal Color: {ann.signal_color}"
            )
        else:
            print(
                f"Class: {ann.class_name}, ID: {ann.class_id}, BBox: {ann.bbox.to_coco()}, "
                f"Occlusion: {ann.occlusion}, Truncation: {ann.truncation}"
            )
