# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

from src.task2.dataset import make_coco_dicts, get_transform

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog,
)
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from src.task2.visualization import TopKLossVisualizationHook


class BDDTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=get_transform(is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=get_transform(is_train=False)
        )


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

    vis_hook = TopKLossVisualizationHook(
        cfg=cfg,
        dataset_name=cfg.DATASETS.TEST[0],
        eval_period=len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH,
        topk=5,
    )
    trainer.register_hooks([vis_hook])
    trainer.resume_or_load(resume=False)
    trainer.train()
