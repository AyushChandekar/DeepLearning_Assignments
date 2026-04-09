"""
Main runner — trains and evaluates all 4 YOLO tasks sequentially.
Run individual parts with:  python part_a_detection.py  (etc.)
Run deployment with:        python deployment/app.py
"""

import sys
import torch

from part_a_detection import train as train_detect, test as test_detect, inference as infer_detect
from part_b_classification import train as train_cls, test as test_cls, inference as infer_cls
from part_c_pose import train as train_pose, test as test_pose, inference as infer_pose
from part_d_obb import train as train_obb, test as test_obb, inference as infer_obb


def show_system_info():
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Python:  {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)


def run_part(name, train_fn, test_fn, infer_fn):
    print(f"\n{'#' * 60}")
    print(f"# {name}")
    print(f"{'#' * 60}")
    best = train_fn()
    test_fn(best)
    infer_fn(best)
    print(f"\n{name} — DONE\n")


if __name__ == "__main__":
    show_system_info()

    run_part("PART A: Object Detection", train_detect, test_detect, infer_detect)
    run_part("PART B: Classification", train_cls, test_cls, infer_cls)
    run_part("PART C: Pose Estimation", train_pose, test_pose, infer_pose)
    run_part("PART D: OBB Detection", train_obb, test_obb, infer_obb)

    print("\n" + "=" * 60)
    print("ALL PARTS COMPLETE!")
    print("To launch the web deployment, run:")
    print("  python deployment/app.py")
    print("Then open http://localhost:5000 in your browser.")
    print("=" * 60)
