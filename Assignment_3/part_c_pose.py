"""
Part C: Pose Estimation using YOLOv8 Nano
- Trains on a Roboflow pose/keypoints dataset
- Runs inference and saves results
"""

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "pose")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def train():
    """Train YOLOv8n-pose model."""
    print("=" * 60)
    print("PART C: Pose Estimation Training")
    print("=" * 60)

    model = YOLO("yolov8n-pose.pt")  # Nano pose model

    results = model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        device="0" if torch.cuda.is_available() else "cpu",
        project=os.path.join(BASE_DIR, "runs"),
        name="pose",
        patience=10,
        workers=2,
    )

    print("\nTraining complete!")
    print(f"Best model: {model.trainer.best}")
    return model.trainer.best


def test(model_path=None):
    """Run validation on the trained model."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "pose", "weights", "best.pt")

    print("\n" + "=" * 60)
    print("PART C: Pose Estimation — Validation")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, device="0" if torch.cuda.is_available() else "cpu")

    print(f"\nmAP50 (box):  {metrics.box.map50:.4f}")
    print(f"mAP50 (pose): {metrics.pose.map50:.4f}")
    return metrics


def inference(model_path=None, source=None):
    """Run inference on sample images."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "runs", "pose", "weights", "best.pt")
    if source is None:
        source = os.path.join(DATASET_DIR, "test", "images")

    print("\n" + "=" * 60)
    print("PART C: Pose Estimation — Inference")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=os.path.join(BASE_DIR, "runs"),
        name="pose_inference",
        device="0" if torch.cuda.is_available() else "cpu",
        conf=0.25,
    )

    for r in results[:5]:
        n_persons = len(r.keypoints) if r.keypoints is not None else 0
        print(f"  {os.path.basename(r.path)}: {n_persons} pose(s) detected")
    return results


if __name__ == "__main__":
    best_model = train()
    test(best_model)
    inference(best_model)
