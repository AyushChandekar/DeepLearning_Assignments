"""
Part A: Object Detection using YOLOv8 Nano
- Trains on a Roboflow detection dataset
- Runs inference and saves results
"""

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "detection")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def train():
    """Train YOLOv8n detection model."""
    print("=" * 60)
    print("PART A: Object Detection Training")
    print("=" * 60)

    model = YOLO("yolov8n.pt")  # Nano model — lightweight for laptop GPU

    results = model.train(
        data=DATA_YAML,
        epochs=25,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        project=os.path.join(BASE_DIR, "runs"),
        name="detect",
        patience=10,
        workers=2,
    )

    print("\nTraining complete!")
    print(f"Best model: {model.trainer.best}")
    return model.trainer.best


def test(model_path=None):
    """Run validation on the trained model."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "detect", "weights", "best.pt")

    print("\n" + "=" * 60)
    print("PART A: Object Detection — Validation")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, device="0" if torch.cuda.is_available() else "cpu")

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


def inference(model_path=None, source=None):
    """Run inference on sample images."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "detect", "weights", "best.pt")
    if source is None:
        source = os.path.join(DATASET_DIR, "test", "images")

    print("\n" + "=" * 60)
    print("PART A: Object Detection — Inference")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=os.path.join(BASE_DIR, "runs"),
        name="detect_inference",
        device="0" if torch.cuda.is_available() else "cpu",
        conf=0.25,
    )

    for r in results[:5]:
        print(f"  {os.path.basename(r.path)}: {len(r.boxes)} detections")
    return results


if __name__ == "__main__":
    best_model = train()
    test(best_model)
    inference(best_model)
