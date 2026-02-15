# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(num_classes: int, checkpoint_path: str = None) -> FasterRCNN:
    """
    Load Faster R-CNN model
    Args:
        num_classes (int): Number of classes (including background)
        checkpoint_path (str): Path to checkpoint file (optional)
    Returns:
        FasterRCNN: Faster R-CNN model
    """
    # Load the model with pretrained weights
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    NUM_CLASSES = 11  # 10 classes + background
    CHECKPOINT = "/home/yaswanth/bosch/code/bdd-detection-bench/output/model_final.pth"  # Set to None for pretrained only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(NUM_CLASSES)
    model.to(device)

    # Training mode syntax:
    model.train()
    images = [torch.randn(3, 800, 800).to(device)]  # List of 3D tensors
    targets = [
        {
            "boxes": torch.tensor(
                [[50, 50, 200, 200], [300, 300, 400, 450]], dtype=torch.float32
            ).to(device),  # [N, 4] in [x1, y1, x2, y2]
            "labels": torch.tensor([1, 3], dtype=torch.int64).to(
                device
            ),  # [N] class labels
        }
    ]

    loss_dict = model(images, targets)  # Returns dict of losses
    losses = sum(loss for loss in loss_dict.values())
    print("Training mode - Loss dict:", loss_dict)
    print("Total loss:", losses.item())

    # Eval mode syntax:
    model.eval()
    with torch.no_grad():
        predictions = model(
            images
        )  # Returns list of dicts with 'boxes', 'labels', 'scores'
    print("Eval mode - Predictions:", predictions[0].keys())
    print("Predicted boxes:", predictions[0]["boxes"])
