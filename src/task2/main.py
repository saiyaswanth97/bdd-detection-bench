# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import cv2
import torch
from torch.cuda.amp import autocast
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
            self.visualize_topk_predictions()

    def _visualize_gt_pred_clean(self, inp, output, metadata):
        img = cv2.imread(inp["file_name"])
        if img is None:
            raise FileNotFoundError(f"Image not found: {inp['file_name']}")
        img_rgb = img[:, :, ::-1]

        v = Visualizer(img_rgb, metadata=metadata)

        # Draw GT in GREEN
        gt_boxes = inp["instances"].gt_boxes.tensor.numpy()
        gt_classes = inp["instances"].gt_classes.numpy()

        for box, cls in zip(gt_boxes, gt_classes):
            v.draw_box(box, edge_color="green")
            v.draw_text(metadata.thing_classes[cls], box[:2], color="green")

        # Draw Predictions in RED
        pred_instances = output["instances"].to("cpu")
        pred_boxes = pred_instances.pred_boxes.tensor.numpy()
        pred_classes = pred_instances.pred_classes.numpy()
        scores = pred_instances.scores.numpy()

        for box, cls, score in zip(pred_boxes, pred_classes, scores):
            label = f"{metadata.thing_classes[cls]} {score:.2f}"
            v.draw_box(box, edge_color="red")
            v.draw_text(label, box[:2], color="red")

        return v.get_output().get_image()

    def visualize_topk_predictions(self):
        model = self.trainer.model
        model.eval()

        mapper = get_transform(is_train=False, keep_gt=True)
        data_loader = build_detection_test_loader(
            self.cfg, self.dataset_name, mapper=mapper
        )
        metadata = self.trainer.metadata

        # Min-heap for top-k (highest scores)
        top_predictions = []
        # Max-heap for bottom-k (lowest scores) - negate scores
        bottom_predictions = []

        with torch.no_grad(), autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            for inputs in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
                outputs = model(inputs)

                for inp, output in zip(inputs, outputs):
                    instances = output["instances"].to("cpu")
                    if len(instances) == 0:
                        continue

                    scores = instances.scores.numpy()

                    for score in scores:
                        score = float(score)

                        # Top-k: maintain min-heap of size k
                        if len(top_predictions) < self.topk:
                            heapq.heappush(top_predictions, (score, inp, output))
                        elif score > top_predictions[0][0]:
                            heapq.heapreplace(top_predictions, (score, inp, output))

                        # Bottom-k: maintain max-heap (negated) of size k
                        if len(bottom_predictions) < self.topk:
                            heapq.heappush(bottom_predictions, (-score, inp, output))
                        elif score < -bottom_predictions[0][0]:
                            heapq.heapreplace(bottom_predictions, (-score, inp, output))

        storage = self.trainer.storage

        # Sort top predictions in descending order (highest first)
        top_predictions.sort(reverse=True, key=lambda x: x[0])
        for i, (score, inp, output) in enumerate(top_predictions):
            img = self._visualize_gt_pred_clean(inp, output, metadata)
            storage.put_image(f"Top_{i + 1}_score_{score:.3f}", img.transpose(2, 0, 1))

        # Sort bottom predictions in ascending order (lowest first)
        bottom_predictions.sort(key=lambda x: -x[0])
        for i, (neg_score, inp, output) in enumerate(bottom_predictions):
            img = self._visualize_gt_pred_clean(inp, output, metadata)
            storage.put_image(
                f"Bottom_{i + 1}_score_{-neg_score:.3f}", img.transpose(2, 0, 1)
            )

        model.train()


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
    cfg.DATALOADER.NUM_WORKERS = 24

    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.IMS_PER_BATCH = 32
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
    cfg.MODEL.WEIGHTS = "./output/model_0004999.pth"

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
