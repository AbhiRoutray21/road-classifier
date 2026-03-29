"""
=============================================================
  Road Classifier - Training Script
  Model  : MobileNetV2 (Transfer Learning)
  Task   : Binary Classification - Road vs Not Road
  Device : CPU-optimised for Intel Core Ultra 7 155U
=============================================================
"""

import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
#  CONFIG  (edit these if needed)
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"      : "data",          # folder with train/ val/ subfolders
    "model_save"    : "models/road_classifier.pth",
    "batch_size"    : 20,              # keep low for CPU
    "num_epochs"    : 20,
    "learning_rate" : 0.001,
    "image_size"    : 224,
    "num_workers"   : 2,              # CPU data loading threads
    "freeze_layers" : True,           # True = only train classifier head (FAST)
}


# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(image_size):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return {"train": train_transforms, "val": val_transforms}


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
def build_model(freeze_layers=True):
    """MobileNetV2 with custom binary classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_layers:
        # Freeze all layers except the classifier
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head: 1280 → 256 → 2
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),          # 2 classes: road, not_road
    )

    return model


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*55}")
    print(f"  Training on: {device}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  Batch size : {CONFIG['batch_size']}")
    print(f"{'='*55}\n")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("-" * 40)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"  {phase.upper()}", leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loop.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"  {phase.upper():5s} → Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


# ─────────────────────────────────────────────
#  PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history, save_path="models/training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss", color="#2196F3")
    ax1.plot(history["val_loss"],   label="Val Loss",   color="#F44336")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", color="#4CAF50")
    ax2.plot(history["val_acc"],   label="Val Acc",   color="#FF9800")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining curves saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cpu")   # Your laptop uses CPU

    # ── Data ──────────────────────────────────
    tf = get_transforms(CONFIG["image_size"])

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(CONFIG["data_dir"], x),
            transform=tf[x]
        )
        for x in ["train", "val"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size  = CONFIG["batch_size"],
            shuffle     = (x == "train"),
            num_workers = CONFIG["num_workers"],
        )
        for x in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    print(f"\nClasses detected: {class_names}")
    print(f"Train samples  : {len(image_datasets['train'])}")
    print(f"Val samples    : {len(image_datasets['val'])}")

    # Save class names for inference
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)

    # ── Model ─────────────────────────────────
    model = build_model(freeze_layers=CONFIG["freeze_layers"])
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params : {trainable:,} / {total:,}")

    # ── Loss & Optimiser ──────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )

    # ── Train ─────────────────────────────────
    model, history = train_model(
        model, dataloaders, criterion, optimizer,
        CONFIG["num_epochs"], device
    )

    # ── Save ──────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict" : model.state_dict(),
        "class_names"      : class_names,
        "config"           : CONFIG,
    }, CONFIG["model_save"])
    print(f"\nModel saved → {CONFIG['model_save']}")

    # ── Plot ──────────────────────────────────
    plot_history(history)


if __name__ == "__main__":
    main()
