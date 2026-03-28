"""
=============================================================
  Road Classifier - Inference Script
  Usage:
    Single image : python predict.py --image path/to/img.jpg
    Folder       : python predict.py --folder path/to/folder
    Webcam       : python predict.py --webcam
=============================================================
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path="models/road_classifier.pth"):
    checkpoint  = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded → Classes: {class_names}")
    return model, class_names


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
#  PREDICT SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_image(model, class_names, image_path):
    img  = Image.open(image_path).convert("RGB")
    inp  = INFER_TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inp)
        probs   = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    label = class_names[pred_idx.item()]
    confidence = conf.item() * 100

    return label, confidence, probs.numpy()


# ─────────────────────────────────────────────
#  DISPLAY RESULT
# ─────────────────────────────────────────────
def show_result(image_path, label, confidence, class_names, probs):
    img = Image.open(image_path).convert("RGB")

    is_road = label == "road"
    color   = "#4CAF50" if is_road else "#F44336"
    emoji   = "🛣️  ROAD" if is_road else "🚫 NOT ROAD"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Image
    ax1.imshow(img)
    ax1.set_title(f"{emoji}\nConfidence: {confidence:.1f}%",
                  fontsize=13, fontweight="bold",
                  color=color)
    ax1.axis("off")

    # Bar chart
    colors = ["#4CAF50" if c == "road" else "#F44336" for c in class_names]
    bars = ax2.barh(class_names, probs * 100, color=colors, edgecolor="white")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Class Probabilities")
    ax2.grid(axis="x", alpha=0.3)

    for bar, p in zip(bars, probs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{p*100:.1f}%", va="center", fontsize=11)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*40}")
    print(f"  Result     : {emoji}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
#  BATCH FOLDER PREDICTION
# ─────────────────────────────────────────────
def predict_folder(model, class_names, folder_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in os.listdir(folder_path)
              if os.path.splitext(f)[1].lower() in exts]

    if not images:
        print("No images found in folder.")
        return

    print(f"\nPredicting {len(images)} images...\n")
    print(f"{'Image':<40} {'Prediction':<12} {'Confidence':>10}")
    print("-" * 65)

    road_count = 0
    for fname in images:
        path = os.path.join(folder_path, fname)
        label, conf, _ = predict_image(model, class_names, path)
        tag = "🛣️" if label == "road" else "🚫"
        print(f"{fname:<40} {tag} {label:<10} {conf:>9.1f}%")
        if label == "road":
            road_count += 1

    print("-" * 65)
    print(f"\nSummary: {road_count} road  |  {len(images)-road_count} not road  |  Total: {len(images)}")


# ─────────────────────────────────────────────
#  LIVE WEBCAM PREDICTION
# ─────────────────────────────────────────────
def predict_webcam(model, class_names):
    cap = cv2.VideoCapture(0)
    print("\nWebcam started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert for model
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        inp   = INFER_TRANSFORM(pil).unsqueeze(0)

        with torch.no_grad():
            out   = model(inp)
            probs = torch.softmax(out, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        label = class_names[idx.item()]
        conf  = conf.item() * 100

        is_road = label == "road"
        color   = (76, 175, 80) if is_road else (244, 67, 54)   # BGR
        text    = f"{'ROAD' if is_road else 'NOT ROAD'}  {conf:.1f}%"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Road Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Road Classifier Inference")
    parser.add_argument("--model",  default="models/road_classifier.pth")
    parser.add_argument("--image",  type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    args = parser.parse_args()

    model, class_names = load_model(args.model)

    if args.image:
        label, conf, probs = predict_image(model, class_names, args.image)
        show_result(args.image, label, conf, class_names, probs)

    elif args.folder:
        predict_folder(model, class_names, args.folder)

    elif args.webcam:
        predict_webcam(model, class_names)

    else:
        print("Please provide --image, --folder, or --webcam flag.")
        print("Example: python predict.py --image myimage.jpg")


if __name__ == "__main__":
    main()
