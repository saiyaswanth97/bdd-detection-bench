# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import cv2
import torch
import heapq
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from src.task2.dataset import get_transform

from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import HookBase
from torchvision.ops import nms as torch_nms


def apply_nms_to_predictions(
    predictions_dict: Dict[str, Any],
    iou_threshold: float = 0.5,
    min_score: float = 0.05,
) -> List[Dict[str, np.ndarray]]:
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping predictions.

    Uses torchvision's optimized NMS implementation to remove redundant detections
    by iteratively selecting the highest-confidence prediction and suppressing all
    overlapping predictions above the IoU threshold.

    Args:
        predictions_dict (Dict[str, Any]): Detectron2 output dictionary containing "instances" key
        iou_threshold (float): IoU threshold for suppression. Predictions with IoU
                              above this value are removed. Default: 0.75
        min_score (float): Minimum confidence score to keep a prediction. Default: 0.05

    Returns:
        List[Dict[str, np.ndarray]]: List of dictionaries, each containing:
            - "box": Bounding box tensor [x1, y1, x2, y2]
            - "class": Class ID tensor
            - "score": Confidence score tensor
            Returns empty list if no instances found.
    """
    instances = predictions_dict.get("instances", None)
    if instances is None:
        return []

    pred_boxes = instances.pred_boxes.tensor
    pred_classes = instances.pred_classes
    scores = instances.scores

    # Apply torchvision's optimized NMS
    keep_indices = torch_nms(
        pred_boxes,
        scores,
        iou_threshold,
    )

    # Convert to list of prediction dictionaries
    filtered_predictions = []
    for idx in keep_indices:
        if scores[idx] < min_score:
            continue
        filtered_predictions.append(
            {"box": pred_boxes[idx], "class": pred_classes[idx], "score": scores[idx]}
        )

    for pred in filtered_predictions:
        pred["box"] = pred["box"].cpu().numpy()
        pred["class"] = pred["class"].cpu().numpy()
        pred["score"] = pred["score"].cpu().numpy()

    return filtered_predictions


class TopKLossVisualizationHook(HookBase):
    """
    A Detectron2 training hook that visualizes the top-k best and worst predictions.

    This hook periodically evaluates the model on a validation set, identifies samples
    with highest and lowest losses, and logs visualizations comparing ground truth
    annotations (green) with model predictions (red) to TensorBoard.

    Attributes:
        cfg: Detectron2 configuration object
        dataset_name: Name of the dataset to evaluate on
        topk: Number of best/worst samples to visualize
        eval_period: How often (in iterations) to perform visualization
    """

    def __init__(
        self,
        cfg: CfgNode,
        dataset_name: str,
        eval_period: int,
        topk: int = 5,
        num_fixed_samples: int = 5,
    ) -> None:
        """
        Initialize the visualization hook.

        Args:
            cfg (CfgNode): Detectron2 configuration object
            dataset_name (str): Name of the registered dataset to visualize
            eval_period (int): Visualization frequency in training iterations
            topk (int): Number of top/bottom samples to select based on loss
            num_fixed_samples (int): Number of random samples to track for progression visualization
        """
        super().__init__()
        self.cfg: CfgNode = cfg.clone()
        self.dataset_name: str = dataset_name
        self.topk: int = topk
        self.eval_period: int = eval_period
        self.num_fixed_samples: int = num_fixed_samples
        self.fixed_samples: Optional[List[List[Dict[str, Any]]]] = None

    def after_step(self) -> None:
        """
        Called after each training step. Triggers visualization at specified intervals.

        Evaluates the model and logs visualizations of best and worst predictions
        when the current iteration is a multiple of eval_period. Also visualizes
        fixed random samples to track model progression over time.
        """
        if (self.trainer.iter + 1) % self.eval_period == 0:
            # Sample fixed samples on first visualization call
            if self.fixed_samples is None:
                self.fixed_samples = self._sample_random_fixed_samples()

            best_samples, worst_samples = self.select_top_k_samples_by_loss()
            self.visualize_and_log_samples(best_samples, label="Best")
            self.visualize_and_log_samples(worst_samples, label="Worst")
            self.visualize_fixed_samples(label="Fixed")

    def _sample_random_fixed_samples(self) -> List[List[Dict[str, Any]]]:
        """
        Sample random samples from the validation set for tracking progression.

        These samples will be visualized at each evaluation period to show
        how model predictions improve over time on the same images.

        Returns:
            List[List[Dict[str, Any]]]: List of randomly sampled input batches
        """
        mapper = get_transform(is_train=False, keep_gt=True)
        data_loader = build_detection_test_loader(
            self.cfg, self.dataset_name, mapper=mapper
        )

        random_samples_id = random.sample(
            range(len(data_loader.dataset)), self.num_fixed_samples
        )

        sampled_inputs = []
        # Collect samples from the dataset
        for idx, inputs in enumerate(data_loader):
            if idx in random_samples_id:
                sampled_inputs.append(inputs)

        return sampled_inputs

    def select_top_k_samples_by_loss(
        self,
    ) -> Tuple[
        List[Tuple[float, List[Dict[str, Any]]]],
        List[Tuple[float, List[Dict[str, Any]]]],
    ]:
        """
        Select top-k samples with highest and lowest losses from the validation set.

        Uses min/max heaps to efficiently track the k-best (lowest loss) and k-worst
        (highest loss) predictions during a single pass through the validation data.
        Only evaluates the first 100 batches for efficiency.

        Returns:
            Tuple[List[Tuple[float, List[Dict[str, Any]]]], List[Tuple[float, List[Dict[str, Any]]]]]:
                Two lists of (loss, inputs) tuples:
                    - best_samples: k samples with lowest loss (sorted ascending)
                    - worst_samples: k samples with highest loss (sorted descending)
        """
        model = self.trainer.model

        best_k_heap = []  # Min-heap to track k samples with highest loss
        worst_k_heap = []  # Max-heap (negated) to track k samples with lowest loss

        mapper = get_transform(is_train=False, keep_gt=True)
        data_loader = build_detection_test_loader(
            self.cfg, self.dataset_name, mapper=mapper
        )
        batch_idx = 0

        with torch.no_grad():
            for inputs in tqdm(
                data_loader, total=len(data_loader), desc="Selecting samples"
            ):
                loss_dict = model(inputs)
                total_loss = float(sum(loss_dict.values()).item())

                # Track k-worst (highest loss) samples using min-heap
                if len(best_k_heap) < self.topk:
                    heapq.heappush(best_k_heap, (total_loss, inputs))
                elif total_loss > best_k_heap[0][0]:
                    heapq.heapreplace(best_k_heap, (total_loss, inputs))

                # Track k-best (lowest loss) samples using max-heap (negated values)
                if len(worst_k_heap) < self.topk:
                    heapq.heappush(worst_k_heap, (-total_loss, inputs))
                elif total_loss < -worst_k_heap[0][0]:
                    heapq.heapreplace(worst_k_heap, (-total_loss, inputs))

                batch_idx += 1
                if batch_idx == 100:
                    break

        # Sort for consistent ordering: worst (high loss) descending, best (low loss) ascending
        best_k_heap.sort(reverse=True, key=lambda x: x[0])
        worst_k_heap.sort(key=lambda x: -x[0])

        return best_k_heap, worst_k_heap

    def visualize_and_log_samples(
        self,
        samples: List[Tuple[float, List[Dict[str, Any]]]],
        label: str = "Best",
    ) -> None:
        """
        Generate and log visualizations for a set of samples to TensorBoard.

        For each sample, creates an image overlay showing ground truth boxes (green)
        and model predictions (red) with confidence scores and class labels.

        Args:
            samples (List[Tuple[float, List[Dict[str, Any]],]]):
                List of (loss, inputs) tuples to visualize
            label (str): Prefix for the logged images
        """
        model = self.trainer.model
        model.eval()

        storage = self.trainer.storage
        metadata = self.trainer.metadata

        with torch.no_grad():
            for i, sample in enumerate(samples):
                loss, input_data = sample
                predictions = model(input_data)

                # Generate visualization with GT and predictions
                visualization_img = self._create_detection_visualization(
                    input_data[0], predictions[0], metadata, loss
                )
                storage.put_image(
                    f"{label}_{i + 1}", visualization_img.transpose(2, 0, 1)
                )

        model.train()

    def visualize_fixed_samples(self, label: str = "Fixed") -> None:
        """
        Visualize the fixed random samples to track model progression over time.

        Uses the same randomly sampled images at each evaluation period to show
        how predictions improve throughout training. This provides a consistent
        view of model learning progress.

        Args:
            label (str): Prefix for the logged images
        """
        if self.fixed_samples is None:
            return

        model = self.trainer.model
        model.eval()

        storage = self.trainer.storage
        metadata = self.trainer.metadata

        with torch.no_grad():
            for i, input_data in enumerate(self.fixed_samples):
                # Compute current loss and predictions
                predictions = model(input_data)

                # Generate visualization with GT and predictions
                visualization_img = self._create_detection_visualization(
                    input_data[0], predictions[0], metadata, None
                )
                storage.put_image(
                    f"{label}_{i + 1}", visualization_img.transpose(2, 0, 1)
                )

        model.train()

    def _create_detection_visualization(
        self,
        input_dict: Dict[str, Any],
        predictions: Dict[str, Any],
        metadata: Metadata,
        loss_value: float = None,
    ) -> np.ndarray:
        """
        Create a visualization image with ground truth and predicted bounding boxes.

        Ground truth boxes are drawn in green with class labels.
        Predicted boxes are drawn in red with class labels and confidence scores.
        The total loss is displayed in the top-left corner.

        Args:
            input_dict (Dict[str, Any]): Input data dictionary containing image path and GT instances
            predictions (Dict[str, Any]): Model prediction output dictionary with instances
            metadata (Metadata): Dataset metadata containing class names
            loss_value (float): Total loss value for this sample

        Returns:
            np.ndarray: RGB visualization image with overlaid boxes and labels

        Raises:
            FileNotFoundError: If the image file cannot be loaded
        """
        # Load and convert image to RGB
        img_bgr = cv2.imread(input_dict["file_name"])
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {input_dict['file_name']}")
        img_rgb = img_bgr[:, :, ::-1]

        visualizer = Visualizer(img_rgb, metadata=metadata)

        # Apply NMS to filter overlapping predictions
        filtered_predictions = apply_nms_to_predictions(predictions)

        # Draw ground truth boxes in GREEN
        gt_boxes = input_dict["instances"].gt_boxes.tensor.numpy()
        gt_classes = input_dict["instances"].gt_classes.numpy()

        for box, class_id in zip(gt_boxes, gt_classes):
            visualizer.draw_box(box, edge_color="green")
            visualizer.draw_text(
                metadata.thing_classes[class_id], box[:2], color="green"
            )

        # Draw predicted boxes in RED
        if filtered_predictions:
            pred_boxes = [pred["box"] for pred in filtered_predictions]
            pred_classes = [pred["class"] for pred in filtered_predictions]
            pred_scores = [pred["score"] for pred in filtered_predictions]

            for box, class_id, confidence in zip(pred_boxes, pred_classes, pred_scores):
                label = f"{metadata.thing_classes[class_id]} {confidence:.2f}"
                visualizer.draw_box(box, edge_color="red")
                visualizer.draw_text(label, box[:2], color="red")

        # Get the final visualization and add loss text
        output_img = visualizer.get_output().get_image()
        if loss_value is not None:
            cv2.putText(
                output_img,
                f"Loss: {loss_value:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        return output_img
