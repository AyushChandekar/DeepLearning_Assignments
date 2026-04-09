"""
Part D: Oriented Bounding Box (OBB) Detection using YOLOv8 Nano
- Trains on a Roboflow OBB/rotated dataset
- Runs inference and saves results
"""

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "obb")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def train():
    """Train YOLOv8n-obb model."""
    print("=" * 60)
    print("PART D: OBB Detection Training")
    print("=" * 60)

    model = YOLO("yolov8n-obb.pt")  # Nano OBB model

    results = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        project=os.path.join(BASE_DIR, "runs"),
        name="obb",
        patience=10,
        workers=2,
    )

    print("\nTraining complete!")
    print(f"Best model: {model.trainer.best}")
    return model.trainer.best


def test(model_path=None):
    """Run validation on the trained model."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "obb", "weights", "best.pt")

    print("\n" + "=" * 60)
    print("PART D: OBB Detection — Validation")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, device="0" if torch.cuda.is_available() else "cpu")

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


def inference(model_path=None, source=None):
    """Run inference on sample images."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "obb", "weights", "best.pt")
    if source is None:
        source = os.path.join(DATASET_DIR, "test", "images")

    print("\n" + "=" * 60)
    print("PART D: OBB Detection — Inference")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=os.path.join(BASE_DIR, "runs"),
        name="obb_inference",
        device="0" if torch.cuda.is_available() else "cpu",
        conf=0.25,
    )

    for r in results[:5]:
        n_obbs = len(r.obb) if r.obb is not None else 0
        print(f"  {os.path.basename(r.path)}: {n_obbs} oriented boxes detected")
    return results


if __name__ == "__main__":
    best_model = train()
    test(best_model)
    inference(best_model)
