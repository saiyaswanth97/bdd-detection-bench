# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Torch Dataset for loading and preprocessing data for training and evaluation."""

import os
import torch
import torchvision
from PIL import Image
from typing import List, Tuple
from src.utils.parse_annotations import parse_annotations
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS


class BddDataset:
    """Custom dataset class for loading images and annotations from the BDD100K dataset."""

    def __init__(self, image_folder: str, annotation_folder: str):
        """Initialize the dataset with paths to image and annotation folders.
        Args:
            image_folder (str): Path to the folder containing images.
            annotation_folder (str): Path to the folder containing annotation JSON files.
        Raises:
            FileNotFoundError: If the image folder or annotation folder does not exist.
        """
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder

        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
        if not os.path.exists(annotation_folder):
            raise FileNotFoundError(
                f"Annotation folder {annotation_folder} does not exist."
            )

        self.data = self.load_annotations()
        self.augmentations = None  # Placeholder for any augmentations you want to apply

    def load_annotations(self) -> List[Tuple[str, List[dict]]]:
        """
        Load annotations from the annotation folder and return a list of tuples (image_path, annotations).
        Returns:
            List[Tuple[str, List[dict]]]:
                - image_path (str): Path to the image file.
                - annotations (List[dict]): List of annotation dictionaries for the image.
        """
        annotations = []
        for filename in os.listdir(self.annotation_folder):
            if filename.endswith(".json"):
                annotation_path = os.path.join(self.annotation_folder, filename)
                # TODO: move away from cutom datatype
                image_name, annots = parse_annotations(annotation_path)
                # Add .jpg extension to the image name
                image_path = os.path.join(self.image_folder, image_name + ".jpg")
                if os.path.exists(image_path):
                    annotations.append((image_path, annots))
        return annotations

    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from the given path and convert it to a tensor.
        Args:
            image_path (str): Path to the image file.
        Returns:
            torch.Tensor: A tensor representation of the image.
        """
        img = Image.open(image_path).convert("RGB")
        # TODO: Add augmentation and preprocessing here if needed
        input = torchvision.transforms.ToTensor()(img)
        return input

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, DetDataSample]:
        """Get a sample from the dataset at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, DetDataSample]:
                - image (torch.Tensor): The image tensor.
                - data_sample (DetDataSample): The corresponding data sample containing annotations.
        """
        image_path, annots = self.data[idx]
        data_sample = DetDataSample()
        instances = InstanceData()
        bboxes = []
        labels = []
        for ann in annots:
            bboxes.append([ann.bbox.x1, ann.bbox.y1, ann.bbox.x2, ann.bbox.y2])
            labels.append(ann.class_id)
        instances.bboxes = bboxes
        instances.labels = labels
        data_sample.gt_instances = instances

        image = self.load_image(image_path)
        return image, data_sample

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, DetDataSample]],
    ) -> Tuple[torch.Tensor, List[DetDataSample]]:
        """
        Custom collate function to handle batches of images and annotations.
        Args:
            batch (List[Tuple[torch.Tensor, DetDataSample]]): A list of tuples containing images and data samples.
        Returns:
            Tuple[torch.Tensor, List[DetDataSample]]:
                - images (torch.Tensor): A batch of images stacked into a single tensor.
                - data_samples (List[DetDataSample]): A list of data samples corresponding to each image in the batch.
        """
        images, data_samples = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, data_samples


@DATASETS.register_module()
class BddDatasetEval(BaseDetDataset):
    def __init__(self, image_folder: str, annotation_folder: str):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder

    def load_data_list(self) -> List[dict]:
        label_files = os.listdir(self.annotation_folder)
        data_list = []

        for label_file in label_files:
            image_name, label = parse_annotations(
                os.path.join(self.annotation_folder, label_file)
            )
            image_path = os.path.join(self.image_folder, image_name + ".jpg")

            instances = []
            for instance in label:
                instances.append(
                    {
                        "bbox": [
                            instance.bbox.x1,
                            instance.bbox.y1,
                            instance.bbox.x2,
                            instance.bbox.y2,
                        ],
                        "label": instance.class_id,
                    }
                )

            data_list.append(
                {
                    "image_path": image_path,
                    "instances": instances,
                    "image_id": image_name,
                }
            )

        return data_list


if __name__ == "__main__":
    IMG_DIR = "data/bdd100k_images_100k/100k/test/"
    ANN_DIR = "data/bdd100k_labels/100k/test/"

    dataset = BddDataset(IMG_DIR, ANN_DIR)
    print(f"Dataset size: {len(dataset)}")

    for i in range(3):
        img, data_sample = dataset[i]
        print(
            f"Image shape: {img.shape}, Number of annotations: {len(data_sample.gt_instances.bboxes)}"
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=dataset.collate_fn
    )
    for batch_images, batch_data_samples in data_loader:
        print(
            f"Batch images shape: {batch_images.shape}, Batch size: {len(batch_data_samples)}"
        )
        print(
            f"First image shape in batch: {batch_images[0].shape}, First data sample annotations: {len(batch_data_samples[0].gt_instances.bboxes)}"
        )
        print(
            f"Labels for first data sample: {batch_data_samples[0].gt_instances.labels}"
        )
        print(
            f"BBoxes for first data sample: {batch_data_samples[0].gt_instances.bboxes}"
        )
        break  # Just test one batch

    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="RandomResize", scale=[(1333, 640), (1333, 800)], keep_ratio=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(type="PackDetInputs"),
    ]

    train_dataloader = dict(
        batch_size=2,
        num_workers=2,
        dataset=dict(
            type="BddDatasetEval",
            image_folder=IMG_DIR,
            annotation_folder=ANN_DIR,
            pipeline=train_pipeline,
        ),
        collate_fn=dict(type="default_collate"),
    )

    dataset = DATASETS.build(train_dataloader["dataset"])
    print(f"MMDet Dataset size: {len(dataset)}")
    print(f"First data sample: {dataset[0]}")
