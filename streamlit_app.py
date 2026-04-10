"""
Streamlit Deployment - Multi-Task YOLO Vision System
Serves all 4 models (Detection, Classification, Pose, OBB).
"""

import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "Assignment_3", "runs")

MODEL_PATHS = {
    "Object Detection": os.path.join(RUNS_DIR, "detect2", "weights", "best.pt"),
    "Image Classification": os.path.join(RUNS_DIR, "classify", "weights", "best.pt"),
    "Pose Estimation": os.path.join(RUNS_DIR, "pose", "weights", "best.pt"),
    "Oriented Bounding Box (OBB)": os.path.join(RUNS_DIR, "obb", "weights", "best.pt"),
}

TASK_INFO = {
    "Object Detection": {
        "icon": "🎯",
        "desc": "Detect and localize vehicles in images with bounding boxes.",
        "model": "YOLOv8n",
        "classes": "12 vehicle types (car, bus, truck, etc.)",
    },
    "Image Classification": {
        "icon": "🏷️",
        "desc": "Classify cropped vehicle images into one of 12 categories.",
        "model": "YOLOv8n-cls",
        "classes": "12 vehicle classes",
    },
    "Pose Estimation": {
        "icon": "🦴",
        "desc": "Detect human body keypoints (17-point COCO pose format).",
        "model": "YOLOv8n-pose",
        "classes": "17 body keypoints",
    },
    "Oriented Bounding Box (OBB)": {
        "icon": "📐",
        "desc": "Detect vehicles with rotation-aware oriented bounding boxes.",
        "model": "YOLOv8n-obb",
        "classes": "12 vehicle types",
    },
}


# ── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_model(task: str) -> YOLO:
    """Load a YOLO model and cache it across reruns."""
    path = MODEL_PATHS[task]
    if not os.path.exists(path):
        st.error(f"Model not found at `{path}`. Train the model first.")
        st.stop()
    return YOLO(path)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Task YOLO Vision System",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6c3fc0, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
    }
    .task-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .task-card:hover { border-color: #8b5cf6; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6c3fc0, #8b5cf6);
        color: white;
        border: none;
        padding: 0.7rem;
        font-size: 1.1rem;
        border-radius: 10px;
    }
    .stButton > button:hover { opacity: 0.9; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #444;
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header"><h1>Multi-Task YOLO Vision System</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#aaa;'>"
    "Object Detection &bull; Classification &bull; Pose Estimation &bull; OBB"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    task = st.selectbox(
        "Select Task",
        list(MODEL_PATHS.keys()),
        format_func=lambda t: f"{TASK_INFO[t]['icon']}  {t}",
    )

    info = TASK_INFO[task]
    st.markdown(f"**Model:** `{info['model']}`")
    st.markdown(f"**Classes:** {info['classes']}")
    st.markdown(f"_{info['desc']}_")

    st.divider()
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    st.divider()
    st.markdown("##### Available Models")
    for t, path in MODEL_PATHS.items():
        status = "✅" if os.path.exists(path) else "❌"
        st.markdown(f"{status} {TASK_INFO[t]['icon']} {t}")

    st.divider()
    st.caption("Built with YOLOv8 + Streamlit")
    st.caption("Ayush Chandekar | Deep Learning Sem-6")

# ── Main content ─────────────────────────────────────────────────────────────
col_upload, col_result = st.columns(2)

with col_upload:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_result:
    st.subheader("Results")

    if uploaded_file is not None:
        run_btn = st.button("Run Inference", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Running inference..."):
                model = load_model(task)
                results = model.predict(source=image, conf=conf_threshold, device="cpu")
                result = results[0]

                # Annotated image
                annotated = result.plot()
                annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB
                st.image(annotated_rgb, caption="Prediction", use_container_width=True)

                # Task-specific details
                st.divider()

                if task == "Object Detection":
                    n = len(result.boxes)
                    st.metric("Objects Detected", n)
                    if n > 0:
                        st.markdown("**Detections:**")
                        for cls_id, conf in zip(
                            result.boxes.cls.tolist(), result.boxes.conf.tolist()
                        ):
                            name = result.names[int(cls_id)]
                            st.markdown(f"- **{name}** — `{conf:.1%}`")

                elif task == "Image Classification":
                    top1_idx = result.probs.top1
                    top1_conf = result.probs.top1conf.item()
                    st.metric("Prediction", result.names[top1_idx], f"{top1_conf:.1%}")

                    st.markdown("**Top-5 Predictions:**")
                    for i in result.probs.top5:
                        name = result.names[i]
                        conf = result.probs.data[i].item()
                        st.progress(conf, text=f"{name}: {conf:.1%}")

                elif task == "Pose Estimation":
                    n = len(result.keypoints) if result.keypoints is not None else 0
                    st.metric("Poses Detected", n)

                elif task == "Oriented Bounding Box (OBB)":
                    n = len(result.obb) if result.obb is not None else 0
                    st.metric("Oriented Boxes Detected", n)
                    if n > 0:
                        st.markdown("**Detections:**")
                        for cls_id, conf in zip(
                            result.obb.cls.tolist(), result.obb.conf.tolist()
                        ):
                            name = result.names[int(cls_id)]
                            st.markdown(f"- **{name}** — `{conf:.1%}`")
    else:
        st.info("Upload an image on the left and click **Run Inference** to see results.")
