"""
Part B: Image Classification using YOLOv8 Nano
- Trains on a Roboflow classification dataset (folder format)
- Runs inference and saves results
"""

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "classification")


def train():
    """Train YOLOv8n-cls classification model."""
    print("=" * 60)
    print("PART B: Image Classification Training")
    print("=" * 60)

    model = YOLO("yolov8n-cls.pt")  # Nano classification model

    results = model.train(
        data=DATASET_DIR,
        epochs=25,
        imgsz=224,
        batch=32,
        device="0" if torch.cuda.is_available() else "cpu",
        project=os.path.join(BASE_DIR, "runs"),
        name="classify",
        patience=10,
        workers=2,
    )

    print("\nTraining complete!")
    print(f"Best model: {model.trainer.best}")
    return model.trainer.best


def test(model_path=None):
    """Run validation on the trained model."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "classify", "weights", "best.pt")

    print("\n" + "=" * 60)
    print("PART B: Image Classification — Validation")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=DATASET_DIR, device="0" if torch.cuda.is_available() else "cpu")

    print(f"\nTop-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    return metrics


def inference(model_path=None, source=None):
    """Run inference on sample images."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "classify", "weights", "best.pt")
    if source is None:
        source = os.path.join(DATASET_DIR, "test")

    print("\n" + "=" * 60)
    print("PART B: Image Classification — Inference")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=os.path.join(BASE_DIR, "runs"),
        name="classify_inference",
        device="0" if torch.cuda.is_available() else "cpu",
    )

    for r in results[:5]:
        top1_idx = r.probs.top1
        top1_conf = r.probs.top1conf.item()
        class_name = r.names[top1_idx]
        print(f"  {os.path.basename(r.path)}: {class_name} ({top1_conf:.2%})")
    return results


if __name__ == "__main__":
    best_model = train()
    test(best_model)
    inference(best_model)
