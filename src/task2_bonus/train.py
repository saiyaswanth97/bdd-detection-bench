# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.task2_bonus import load_model, BDDDataset
from src.task2_bonus import calculate_ap_all_classes
from torchvision.ops import nms


def train(model, dataloader, optimizer, device, epoch, scaler, writer):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            loss_dict = model(images, targets)
            for key, value in loss_dict.items():
                writer.add_scalar(
                    f"Loss/{key}",
                    value.item(),
                    epoch * len(dataloader) + dataloader._index,
                )
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        total_loss += losses.item()

    average_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/average", average_loss, epoch)
    print(f"Epoch {epoch} - Average Loss: {average_loss:.4f}")


def validate(model, dataloader, device, writer):
    model.eval()
    total_mAP = 0
    # Implement validation logic here (e.g., calculate mAP)
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)
            # Apply NMS and calculate metrics here
            for target, output in zip(targets, outputs):
                # Apply NMS using torchvision's implementation
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                # Apply NMS and get indices of kept boxes
                keep_indices = nms(boxes, scores, iou_threshold=0.5)

                # Filter boxes, scores, and labels using the kept indices
                pred_dict = {
                    "boxes": boxes[keep_indices],
                    "labels": labels[keep_indices],
                    "scores": scores[keep_indices],
                }
                AP_class = calculate_ap_all_classes(target, pred_dict)
                mAP = sum(AP_class[class_id][-1] for class_id in AP_class)
                # mAP = mAP.item() if isinstance(mAP, torch.Tensor) else mAP  # Convert to scalar if it's a tensor
                writer.add_scalar(
                    "Validation/mAP", mAP, 0
                )  # Log mAP (last element in list)
                total_mAP += mAP
    average_mAP = total_mAP / len(dataloader)
    writer.add_scalar("Validation/average_mAP", average_mAP, 0)
    print(f"Validation mAP: {average_mAP:.4f}")


if __name__ == "__main__":
    # Config
    NUM_CLASSES = 11
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    LR = 0.005

    TRAIN_ANNO_DIR = "data/bdd100k_labels/100k/train/"
    TRAIN_IMG_DIR = "data/bdd100k_images_100k/100k/train/"
    OUTPUT_DIR = "./output/task2b"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = BDDDataset(TRAIN_ANNO_DIR, TRAIN_IMG_DIR, device, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
    )
    writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/logs")

    # Load model
    model = load_model(NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"Training on {device}")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Training loop
    scaler = GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(model, train_loader, optimizer, device, epoch, scaler, writer)

        # Save checkpoint
        checkpoint_path = f"{OUTPUT_DIR}/checkpoint_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        lr_scheduler.step()

        validate(model, train_loader, device, writer)

    # Save final model
    final_path = f"{OUTPUT_DIR}/model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Model saved to {final_path}")

    writer.flush()
    writer.close()
