# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import torch
import os
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from src.utils.parse_annotations import parse_annotations


class BDDDataset(Dataset):
    def __init__(
        self, annotation_folder: str, img_dir: str, train: bool = False
    ) -> None:
        """
        Custom Dataset for BDD100K object detection.
        Args:
            annotation_folder (str): Path to the folder containing JSON annotation files.
            img_dir (str): Directory containing the images.
            train (bool): Whether to apply training data augmentations.
        """
        self.annotation_folder = annotation_folder
        self.img_dir = img_dir
        self.train = train
        self.annotations = []
        self.data = self.load_annotations()

    def load_annotations(self) -> None:
        # Load the JSON annotations and parse them into a list of dictionaries
        # Each dictionary contains:
        # - 'image_id': The filename of the image
        # - 'annotations': A list of annotations List[torch.tensor, torch.tensot] for that image, where each annotation is a dictionary with:
        #   - 'bbox': [x1, y1, x2, y2] (Nx4) tensor of bounding box coordinates in COCO format
        #   - 'category_id': The class ID of the object (N) tensor of class labels

        self.annotations = []
        total_invalid = 0
        for file in os.listdir(self.annotation_folder):
            if file.endswith(".json"):
                annotation_json = os.path.join(self.annotation_folder, file)

            name, annotations = parse_annotations(annotation_json)

            bboxes = []
            classes = []

            for annotation in annotations:
                x, y, w, h = annotation.bbox.to_coco()
                if w > 0 and h > 0:
                    bboxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
                    classes.append(annotation.class_id)
                else:
                    total_invalid += 1

            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.int64)

            self.annotations.append(
                {"image_id": name, "bboxes": bboxes, "classes": classes}
            )
        print(
            f"Loaded {len(self.annotations)} annotations with {total_invalid} invalid bounding boxes."
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a sample from the dataset. Applies basic augmentation.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, dict]: (image, target) where:
                - image (torch.Tensor): The image tensor of shape [3, H, W].
                - target (dict): A dictionary containing:
                    - 'boxes': Tensor of shape [N, 4] with bounding box coordinates in [x1, y1, x2, y2] format.
                    - 'labels': Tensor of shape [N] with class labels.
        """
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        bboxes = annotation["bboxes"].clone()
        classes = annotation["classes"].clone()

        # Load image
        img_path = f"{self.img_dir}{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.train:
            orig_w, orig_h = image.size

            target_h, target_w = 360, 640
            image = F.resize(image, [target_h, target_w])
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * target_w / orig_w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * target_h / orig_h

            if torch.rand(1) < 0.5:
                image = F.hflip(image)
                bboxes[:, [0, 2]] = target_w - bboxes[:, [2, 0]]
            if torch.rand(1) < 0.5:
                image = F.vflip(image)
                bboxes[:, [1, 3]] = target_h - bboxes[:, [3, 1]]

            image = F.adjust_brightness(image, 1 + (torch.rand(1).item() - 0.5) * 0.4)
            image = F.adjust_contrast(image, 1 + (torch.rand(1).item() - 0.5) * 0.4)

        image = F.to_tensor(image)

        return image, {"boxes": bboxes, "labels": classes}

    def collate_fn(
        self, batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Custom collate function for object detection models (e.g., Faster R-CNN).
        Required because detection models expect:
            images: List[Tensor[C, H, W]]
            targets: List[Dict[str, Tensor]]
        Args:
            batch (List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
                List of tuples where each tuple is (image, target) from __getitem__.
                - image: Tensor of shape [3, H, W]
                - target: Dict with 'boxes' and 'labels' keys
        Returns:
            Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]: A tuple containing:
                - images: List of image tensors
                - targets: List of target dictionaries
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets


if __name__ == "__main__":
    dataset = BDDDataset(
        annotation_folder="data/bdd100k_labels/100k/train/",
        img_dir="data/bdd100k_images_100k/100k/train/",
        train=True,
    )
    print(f"Dataset length: {len(dataset)}")
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Target keys: {target.keys()}")
    print(f"Boxes shape: {target['boxes'].shape}")
    print(f"Labels shape: {target['labels'].shape}")

    for img, tgt in tqdm(dataset):
        bbox_h = tgt["boxes"][:, 3] - tgt["boxes"][:, 1]
        bbox_w = tgt["boxes"][:, 2] - tgt["boxes"][:, 0]

        if (bbox_h <= 0).any() or (bbox_w <= 0).any():
            print(
                f"Found invalid bounding box with non-positive width or height in image {tgt['image_id']}"
            )
