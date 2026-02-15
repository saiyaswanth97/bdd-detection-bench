# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import torch
from torchvision.ops import box_iou
from typing import Tuple, Dict, List
from src.utils.parse_annotations import CLASSES


def calculate_precision_recall_single_class(
    gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate precision and recall for a single class.

    Args:
        gt_boxes (torch.Tensor): List of ground truth bounding boxes [N, 4] in [x1, y1, x2, y2]
        pred_boxes (torch.Tensor): List of predicted bounding boxes [M, 4] in [x1, y1, x2, y2]
                    This is sorted by confidence score in descending order.
        iou_threshold (float): IoU threshold to consider a prediction as True Positive

    Returns:
        Tuple[float, float]: Precision and Recall values for the given IoU threshold
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0

    ious = box_iou(pred_boxes, gt_boxes)  # [M, N]

    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for pred_idx in range(pred_boxes.shape[0]):
        max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)

        if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
            true_positives += 1
            matched_gt.add(gt_idx.item())
        else:
            false_positives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = true_positives / gt_boxes.shape[0]

    return precision, recall


def calculate_ap_per_class(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    iou_range: torch.Tensor | None = None,
) -> List[float]:
    """Calculate Average Precision (AP) for a single class.
    Args:
        gt_boxes (torch.Tensor): Tensor of ground truth bounding boxes [N, 4] in [x1, y1, x2, y2]
        pred_boxes (torch.Tensor): Tensor of predicted bounding boxes [M, 4] in [x1, y1, x2, y2]
                    sorted by confidence score in descending order.
        iou_range (torch.Tensor | None): Range of IoU thresholds to calculate AP over (default: [0.5, 0.55, ..., 0.95])
    Returns:
        List[float]: List of AP values for each IoU threshold, with mAP appended at the end
    """
    if iou_range is None:
        iou_range = torch.arange(0.5, 0.95, 0.05)

    ap_iou = []
    for iou_threshold in iou_range:
        precision, recall = calculate_precision_recall_single_class(
            gt_boxes, pred_boxes, iou_threshold
        )
        ap_iou.append(
            precision * recall
        )  # Simplified AP calculation (not the standard way)

    mAP = sum(ap_iou) / len(ap_iou) if len(ap_iou) > 0 else 0.0
    ap_iou.append(mAP)  # Append mAP at the end for reference

    return ap_iou


def calculate_ap_all_classes(
    gt_boxes_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]
) -> Dict[int, List[float]]:
    """
    Calculate AP & mAP for all classes
    Args:
        gt_boxes_dict (Dict[str, torch.Tensor]): Dict with 'boxes' and 'labels' keys containing ground truth boxes and labels
        pred_dict (Dict[str, torch.Tensor]): Dict with 'boxes', 'labels', and 'scores' keys containing predicted boxes,
                   class labels, and confidence scores (torchvision format)
    Returns:
        Dict[int, List[float]]: Dict mapping class_id to list of AP values per IoU threshold
    """

    metrics = {}
    for class_id in range(1, len(CLASSES)):  # Skip background class (0)
        gt_ids = torch.where(gt_boxes_dict["labels"] == class_id)[0]
        gt_boxes = gt_boxes_dict["boxes"][
            gt_ids
        ].tolist()  # Get GT boxes for this class
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        pred_ids = torch.where(pred_dict["labels"] == class_id)[0]
        pred_boxes = pred_dict["boxes"][pred_ids].tolist()
        pred_scores = pred_dict["scores"][pred_ids].tolist()

        pred_boxes_sorted = sorted(
            zip(pred_boxes, pred_scores), key=lambda x: x[1], reverse=True
        )
        pred_boxes_sorted = [box for box, score in pred_boxes_sorted]
        pred_boxes_sorted = torch.tensor(pred_boxes_sorted, dtype=torch.float32)

        ap_iou = calculate_ap_per_class(gt_boxes, pred_boxes_sorted)
        metrics[class_id] = ap_iou

    return metrics


def apply_nms(
    predictions: List[List[float]], iou_threshold: float = 0.5
) -> List[List[float]]:
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Args:
        predictions (List[List[float]]): List or tensor of predicted bounding boxes [M, 5] in [x1, y1, x2, y2, confidence]
        iou_threshold (float): IoU threshold to suppress boxes

    Returns:
        List[List[float]]: List of filtered boxes after NMS in [x1, y1, x2, y2, confidence] format
    """
    if len(predictions) == 0:
        return []

    predictions = torch.tensor(predictions, dtype=torch.float32)
    predictions = predictions[
        predictions[:, 4].argsort(descending=True)
    ]  # Sort by confidence
    keep_boxes = []

    while predictions.shape[0] > 0:
        best_box = predictions[0].tolist()
        keep_boxes.append(best_box)

        if predictions.shape[0] == 1:
            break

        rest_boxes = predictions[1:]
        ious = box_iou(rest_boxes[:, :4], torch.tensor(best_box[:4]).unsqueeze(0))[:, 0]

        predictions = rest_boxes[ious < iou_threshold]

    return keep_boxes


if __name__ == "__main__":
    print("Running metrics tests...\n")

    # Test 1: calculate_precision_recall_single_class
    print("=" * 60)
    gt_boxes = torch.tensor(
        [[10, 10, 50, 50], [60, 60, 100, 100], [120, 120, 160, 160]],
        dtype=torch.float32,
    )

    pred_boxes = torch.tensor(
        [
            [12, 12, 52, 52],  # Should match first GT (high IoU)
            [61, 61, 101, 101],  # Should match second GT (high IoU)
            [200, 200, 240, 240],  # False positive (no match)
        ],
        dtype=torch.float32,
    )

    precision, recall = calculate_precision_recall_single_class(
        gt_boxes, pred_boxes, iou_threshold=0.5
    )
    print("Test 1: Precision and Recall")
    print(
        f"Precision: {precision:.4f} (expected: 0.667), Recall: {recall:.4f} (expected: 0.667)\n"
    )

    # Test 2: apply_nms
    pred_boxes_with_conf = [
        [10, 10, 50, 50, 0.9],
        [12, 12, 52, 52, 0.8],  # Overlaps with first box, should be suppressed
        [60, 60, 100, 100, 0.85],
        [120, 120, 160, 160, 0.7],
    ]

    nms_result = apply_nms(pred_boxes_with_conf, iou_threshold=0.5)
    print("Test 2: Non-Maximum Suppression")
    print(
        f"Input boxes {len(pred_boxes_with_conf)}, after NMS: {len(nms_result)} (expected: 3)\n"
    )

    # Test 3: calculate_ap_per_class
    gt_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32)

    pred_boxes = torch.tensor(
        [[12, 12, 52, 52], [61, 61, 101, 101]], dtype=torch.float32
    )

    ap_iou_list = calculate_ap_per_class(gt_boxes, pred_boxes)
    print("Test 3: Average Precision per Class")
    print(
        f"AP values for IoU thresholds: {ap_iou_list[:-1]}, mAP: {ap_iou_list[-1]:.4f} (expected mAP: 0.667)\n"
    )
