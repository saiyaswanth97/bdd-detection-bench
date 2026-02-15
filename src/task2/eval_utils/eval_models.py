# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Evaluate a finetuned Detectron2 Faster R-CNN model on BDD100K validation set."""

import json
import argparse
from typing import Tuple, Dict, Any

from src.task2.dataset import make_coco_dicts, get_transform
from src.utils.parse_annotations import CLASSES

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

CLASS_NAMES = CLASSES.keys()


def build_eval_cfg(weights_path: str, score_thresh: float = 0.05) -> CfgNode:
    """
    Build config matching the training setup with evaluation-specific overrides.

    Args:
        weights_path (str): Path to the model checkpoint (.pth file).
        score_thresh (float): Minimum score threshold for predictions.

    Returns:
        CfgNode: Detectron2 CfgNode configured for evaluation.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.DATASETS.TEST = ("bdd_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.IMS_PER_BATCH = 4
    cfg.OUTPUT_DIR = "./output"
    return cfg


def save_precision_matrix(evaluator: COCOEvaluator, output_dir: str) -> None:
    """
    Save the COCO precision matrix to JSON for later analysis.

    Args:
        evaluator (COCOEvaluator): COCOEvaluator instance after inference_on_dataset.
        output_dir (str): Directory to save the precision matrix.
    """
    from pycocotools.cocoeval import COCOeval
    import os

    predictions_file = os.path.join(output_dir, "coco_instances_results.json")
    if os.path.exists(predictions_file) and hasattr(evaluator, "_coco_api"):
        coco_gt = evaluator._coco_api
        coco_dt = coco_gt.loadRes(predictions_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        precision = coco_eval.eval["precision"]
        with open(f"{output_dir}/precision_matrix.json", "w") as f:
            json.dump(
                {
                    "precision": precision.tolist(),
                    "class_names": list(CLASS_NAMES),
                },
                f,
                indent=2,
            )
        print(f"  Precision matrix saved to {output_dir}/precision_matrix.json")
    else:
        print("  WARNING: Could not save precision matrix")


def evaluate(
    cfg: CfgNode, output_dir: str = "./output/eval"
) -> Tuple[Dict[str, Any], COCOEvaluator]:
    """
    Run COCO evaluation on the validation set.

    Args:
        cfg (CfgNode): Detectron2 config node.
        output_dir (str): Directory to save evaluation results.

    Returns:
        Tuple[Dict[str, Any], COCOEvaluator]: Dictionary of evaluation metrics and COCOEvaluator instance.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(
        "bdd_dataset_val",
        tasks=("bbox",),
        distributed=False,
        output_dir=output_dir,
    )
    val_loader = build_detection_test_loader(
        cfg,
        "bdd_dataset_val",
        mapper=get_transform(is_train=False),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=cfg.TEST.IMS_PER_BATCH,
    )
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    save_precision_matrix(evaluator, output_dir)

    return results, evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Detectron2 model on BDD100K")
    parser.add_argument(
        "--weights",
        type=str,
        default="./output/model_final.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.05,
        help="Score threshold for predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/eval",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    setup_logger()

    # Register val dataset
    make_coco_dicts("val")

    cfg = build_eval_cfg(args.weights, args.score_thresh)
    results, evaluator = evaluate(cfg, args.output_dir)

    # Save overall results to JSON
    results_path = f"{args.output_dir}/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")
