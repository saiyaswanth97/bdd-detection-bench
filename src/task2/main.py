# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import cv2
import torch
import heapq
from tqdm import tqdm
from src.task2.dataset import make_coco_dicts, get_transform

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog,
)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, HookBase


class BDDTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=get_transform(is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=get_transform(is_train=False)
        )


class TOPKVisualizationHook(HookBase):
    def __init__(self, cfg, dataset_name, eval_period, topk=5):
        super().__init__()
        self.cfg = cfg.clone()
        self.dataset_name = dataset_name
        self.topk = topk
        self.eval_period = eval_period

    def after_step(self):
        if (self.trainer.iter + 1) % self.eval_period == 0:
            best, worst = self.pick_images_for_visualization()
            self.visualize_sample(best, string_="Best")
            self.visualize_sample(worst, string_="Worst")

    def pick_images_for_visualization(self):
        model = self.trainer.model
        # model.train()

        top_k = []
        bottom_k = []

        mapper = get_transform(is_train=False, keep_gt=True)
        data_loader = build_detection_test_loader(
            self.cfg, self.dataset_name, mapper=mapper
        )

        with torch.no_grad():
            for inputs in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
                loss_dict = model(inputs)
                total_loss = float(sum(loss_dict.values()).item())

                # Best K
                if len(top_k) < self.topk:
                    heapq.heappush(top_k, (total_loss, inputs, loss_dict))
                elif total_loss > top_k[0][0]:
                    heapq.heapreplace(top_k, (total_loss, inputs, loss_dict))

                # Worst K
                if len(bottom_k) < self.topk:
                    heapq.heappush(bottom_k, (-total_loss, inputs, loss_dict))
                elif total_loss < -bottom_k[0][0]:
                    heapq.heapreplace(bottom_k, (-total_loss, inputs, loss_dict))

        top_k.sort(reverse=True, key=lambda x: x[0])  # Sort by loss descending
        bottom_k.sort(key=lambda x: -x[0])

        return top_k, bottom_k

    def visualize_sample(self, dict, string_="Best"):
        model = self.trainer.model
        model.eval()

        storage = self.trainer.storage
        metadata = self.trainer.metadata

        with torch.no_grad():
            for i, sample in enumerate(dict):
                loss, inp, _ = sample
                outputs = model(inp)

                # Visualize the predictions and GT
                img = self._visualize_gt_pred_clean(inp[0], outputs[0], loss, metadata)
                storage.put_image(f"{string_}_{i + 1}", img.transpose(2, 0, 1))

        model.train()

    def _visualize_gt_pred_clean(self, inp, output, score, metadata):
        img = cv2.imread(inp["file_name"])
        if img is None:
            raise FileNotFoundError(f"Image not found: {inp['file_name']}")
        img_rgb = img[:, :, ::-1]

        v = Visualizer(img_rgb, metadata=metadata)

        output = __class__._apply_nms(output)

        # Draw GT in GREEN
        gt_boxes = inp["instances"].gt_boxes.tensor.numpy()
        gt_classes = inp["instances"].gt_classes.numpy()

        for box, cls in zip(gt_boxes, gt_classes):
            v.draw_box(box, edge_color="green")
            v.draw_text(metadata.thing_classes[cls], box[:2], color="green")

        # Draw Predictions in RED
        pred_boxes = torch.stack([pred["box"] for pred in output]).cpu().numpy()
        pred_classes = torch.stack([pred["class"] for pred in output]).cpu().numpy()
        scores = torch.stack([pred["score"] for pred in output]).cpu().numpy()

        for box, cls, score in zip(pred_boxes, pred_classes, scores):
            label = f"{metadata.thing_classes[cls]} {score:.2f}"
            v.draw_box(box, edge_color="red")
            v.draw_text(label, box[:2], color="red")

        output = v.get_output().get_image()
        cv2.putText(
            output,
            f"Loss: {score:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return output

    @staticmethod
    def _compute_iou(box1, box2):
        x_left = torch.max(box1[0], box2[0])
        y_top = torch.max(box1[1], box2[1])
        x_right = torch.min(box1[2], box2[2])
        y_bottom = torch.min(box1[3], box2[3])

        intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(
            y_bottom - y_top, min=0
        )
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        return torch.where(
            union_area > 0, intersection_area / union_area, torch.tensor(0.0)
        )

    @staticmethod
    def _apply_nms(output, iou_threshold: float = 0.5) -> dict:
        instances = output.get("instances", None)
        if instances is None:
            return []

        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes
        scores = instances.scores

        predictions = []
        for box, cls, score in zip(pred_boxes, pred_classes, scores):
            predictions.append({"box": box, "class": cls, "score": score})

        # Sort predictions by score in descending order
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        selected_predictions = []

        while predictions:
            current = predictions.pop(0)
            selected_predictions.append(current)

            predictions = [
                pred
                for pred in predictions
                if __class__._compute_iou(current["box"], pred["box"]) < iou_threshold
            ]
        return selected_predictions


if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    train_dataset = make_coco_dicts("train")
    val_dataset = make_coco_dicts("val")

    cfg.DATASETS.TRAIN = ("bdd_dataset_train",)
    cfg.DATASETS.TEST = ("bdd_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    cfg.SOLVER.AMP.ENABLED = False
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.BASE_LR = 0.02 * cfg.SOLVER.IMS_PER_BATCH / 16
    EPOCHS = 10
    cfg.SOLVER.MAX_ITER = len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH * EPOCHS
    cfg.SOLVER.CHECKPOINT_PERIOD = len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH // 2
    cfg.SOLVER.STEPS = (
        int(cfg.SOLVER.MAX_ITER * 0.6),
        int(cfg.SOLVER.MAX_ITER * 0.8),
    )
    cfg.TEST.EVAL_PERIOD = len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.OUTPUT_DIR = "./output"
    # cfg.MODEL.WEIGHTS = "./output/model_0004999.pth"

    trainer = BDDTrainer(cfg)
    trainer.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    vis_hook = TOPKVisualizationHook(
        cfg=cfg,
        dataset_name=cfg.DATASETS.TEST[0],
        eval_period=len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH,
        topk=5,
    )
    trainer.register_hooks([vis_hook])
    trainer.resume_or_load(resume=False)
    trainer.train()
