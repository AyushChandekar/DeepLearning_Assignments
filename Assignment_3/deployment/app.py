"""
Flask Deployment — Multi-Task YOLO Vision System
Serves all 4 models (Detection, Classification, Pose, OBB) on localhost.
Uses nano/lightweight models for fast inference.
"""

import os
import io
import base64
import torch
from flask import Flask, request, render_template, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# Model paths — adjust if your run folders have different names (e.g. detect2, classify3)
MODEL_PATHS = {
    "detection": os.path.join(RUNS_DIR, "detect", "weights", "best.pt"),
    "classification": os.path.join(RUNS_DIR, "classify", "weights", "best.pt"),
    "pose": os.path.join(RUNS_DIR, "pose", "weights", "best.pt"),
    "obb": os.path.join(RUNS_DIR, "obb", "weights", "best.pt"),
}

# Lazy-load models
models = {}


def get_model(task):
    if task not in models:
        path = MODEL_PATHS[task]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Train the {task} model first."
            )
        models[task] = YOLO(path)
    return models[task]


@app.route("/")
def index():
    available = {task: os.path.exists(path) for task, path in MODEL_PATHS.items()}
    return render_template("index.html", available=available)


@app.route("/predict", methods=["POST"])
def predict():
    task = request.form.get("task")
    if task not in MODEL_PATHS:
        return jsonify({"error": f"Unknown task: {task}"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        model = get_model(task)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    img = Image.open(file.stream).convert("RGB")

    results = model.predict(source=img, conf=0.25, device="0" if torch.cuda.is_available() else "cpu")
    result = results[0]

    # Render annotated image
    annotated = result.plot()  # BGR numpy array
    annotated_rgb = annotated[:, :, ::-1]  # Convert BGR to RGB
    pil_img = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build response details
    details = {}
    if task == "detection":
        details["detections"] = len(result.boxes)
        details["classes"] = [
            {"name": result.names[int(c)], "conf": f"{conf:.2%}"}
            for c, conf in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist()
            )
        ]
    elif task == "classification":
        top1_idx = result.probs.top1
        details["prediction"] = result.names[top1_idx]
        details["confidence"] = f"{result.probs.top1conf.item():.2%}"
        top5 = result.probs.top5
        details["top5"] = [
            {"name": result.names[i], "conf": f"{result.probs.data[i].item():.2%}"}
            for i in top5
        ]
    elif task == "pose":
        n = len(result.keypoints) if result.keypoints is not None else 0
        details["poses_detected"] = n
    elif task == "obb":
        n = len(result.obb) if result.obb is not None else 0
        details["oriented_boxes"] = n

    return jsonify({"image": img_b64, "task": task, "details": details})


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Task YOLO Vision System — Deployment")
    print("Running on http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
