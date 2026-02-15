# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Visualize Faster R-CNN predictions vs ground truth on BDD100K."""

import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt

from src.task2.dataset import make_coco_dicts
from src.utils.parse_annotations import CLASSES

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import batched_nms


def build_eval_cfg(weights_path: str, score_thresh: float = 0.5):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.DATASETS.TEST = ("bdd_dataset_val",)
    cfg.OUTPUT_DIR = "./output"
    return cfg


def evaluate_sample(
    cfg, image, gt, output_path, nms_thresh=0.5, include_classes=None, save_titled=False
):
    if include_classes is None:
        include_classes = [0, 1, 2]  # Default: car, traffic sign, traffic light

    # Get predictions
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Filter specified classes
    class_mask = sum([instances.pred_classes == c for c in include_classes]) > 0
    filtered = instances[class_mask]

    print(f"Total predictions: {len(instances)}")
    print(f"Filtered (car/sign/light): {len(filtered)}")

    # Apply NMS
    if len(filtered) > 0:
        keep = batched_nms(
            filtered.pred_boxes.tensor,
            filtered.scores,
            filtered.pred_classes,
            nms_thresh,
        )
        filtered = filtered[keep]
        print(f"After NMS ({nms_thresh}): {len(filtered)}")

    # Print prediction stats
    class_names = list(CLASSES.keys())
    print("\nPrediction distribution:")
    for i in include_classes:
        count = (filtered.pred_classes == i).sum().item()
        print(f"  {class_names[i]}: {count}")

    # Parse ground truth
    gt_boxes, gt_labels = [], []
    for obj in gt["frames"][0]["objects"]:
        cat = obj["category"]
        if cat in CLASSES and CLASSES[cat] in include_classes:
            box = obj["box2d"]
            gt_boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            gt_labels.append(cat)

    print(f"\nGround truth: {len(gt_boxes)}")
    for i in include_classes:
        count = sum(1 for lbl in gt_labels if CLASSES[lbl] == i)
        print(f"  {class_names[i]}: {count}")

    # Visualize
    img_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    # Draw GT in light red
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (255, 100, 100), 2)
        cv2.putText(
            img_viz,
            f"GT: {label}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 100),
            2,
        )

    # Draw predictions in green
    for box, cls, score in zip(
        filtered.pred_boxes.tensor.numpy(),
        filtered.pred_classes.numpy(),
        filtered.scores.numpy(),
    ):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_viz,
            f"Pred: {class_names[cls]} {score:.2f}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {output_path}")

    # Save with title (optional)
    if save_titled:
        plt.figure(figsize=(16, 9))
        plt.imshow(img_viz)
        plt.axis("off")
        plt.title(
            "GT (Light Red) vs Predictions (Green)", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        titled_path = output_path.replace(".jpg", "_with_title.jpg")
        plt.savefig(titled_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {titled_path}")

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Detectron2 predictions on BDD100K"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./output/model_final.pth",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.5, help="Score threshold"
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Class IDs to visualize (auto-detected from filename if not specified)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Process single image file (basename without .jpg, e.g., 'sample_car')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to output filename (e.g., '_ckpt1')",
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Skip saving titled version"
    )
    args = parser.parse_args()

    make_coco_dicts("val")
    cfg = build_eval_cfg(args.weights, args.score_thresh)

    # Mapping from sample name to class IDs
    # car=0, traffic sign=1, traffic light=2, person=3, truck=4, bus=5, bike=6, rider=7, motor=8, train=9
    SAMPLE_CLASS_MAP = {
        "sample_bike": [6],  # bike
        "sample_car": [0],  # car
        "sample_car_sign_light": [0, 1, 2],  # car, traffic sign, traffic light
        "sample_light": [2],  # traffic light
        "sample_motorcycle": [8],  # motor
        "sample_no_car_sign_light": [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],  # traffic sign, traffic light (no car)
        "sample_person": [3],  # person
        "sample_rider": [7],  # rider
        "sample_sign": [1],  # traffic sign
        "sample_train": [9],  # train
    }

    # Mapping from class name suffix to class IDs (for image_*_CLASS pattern)
    CLASS_NAME_MAP = {
        "bikes": [6],
        "cars": [0],
        "lights": [2],
        "persons": [3],
        "signs": [1],
        "trucks": [4],
        "buses": [5],
        "riders": [7],
        "motors": [8],
        "trains": [9],
    }

    def get_classes_from_filename(basename):
        if basename in SAMPLE_CLASS_MAP:
            return SAMPLE_CLASS_MAP[basename]

        # Handle image_N_CLASS pattern
        parts = basename.split("_")
        if len(parts) >= 3 and parts[0] == "image":
            class_suffix = "_".join(parts[2:])  # Get everything after "image_N_"
            if class_suffix in CLASS_NAME_MAP:
                return CLASS_NAME_MAP[class_suffix]

        return [0, 1, 2]  # default

    # Get all sample and image files
    import glob

    image_dir = "data/viz/src_images"
    json_dir = "data/viz/src_jsons"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process single image or all images
    if args.image:
        # Single image mode
        all_images = [os.path.join(image_dir, f"{args.image}.jpg")]
        if not os.path.exists(all_images[0]):
            print(f"Error: Image not found: {all_images[0]}")
            exit(1)
        print(f"Processing single image: {args.image}\n")
    else:
        # Batch mode - all images
        sample_images = sorted(glob.glob(f"{image_dir}/sample_*.jpg"))
        image_files = sorted(glob.glob(f"{image_dir}/image_*.jpg"))
        all_images = sample_images + image_files
        print(
            f"Found {len(all_images)} images to process ({len(sample_images)} samples + {len(image_files)} image files)\n"
        )

    for img_path in all_images:
        # Get corresponding JSON file
        basename = os.path.basename(img_path).replace(".jpg", "")
        json_path = os.path.join(json_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            print(f"Skipping {basename}: JSON file not found")
            continue

        # Determine classes to visualize
        if args.classes is not None:
            classes = args.classes
        else:
            classes = get_classes_from_filename(basename)

        class_names = list(CLASSES.keys())
        class_labels = [class_names[c] for c in classes]

        print(f"{'=' * 80}")
        print(f"Processing: {basename} (classes: {', '.join(class_labels)})")
        print(f"{'=' * 80}")

        # Load image and ground truth
        image = cv2.imread(img_path)
        with open(json_path) as f:
            gt = json.load(f)

        # Generate output path with optional suffix
        output_filename = f"{basename}{args.output_suffix}_pred.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Process
        evaluate_sample(
            cfg,
            image,
            gt,
            output_path,
            args.nms_thresh,
            classes,
            save_titled=args.no_title,
        )
        print()
