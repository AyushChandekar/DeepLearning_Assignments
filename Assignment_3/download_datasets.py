"""
Download datasets from Roboflow for all 4 tasks.
- Part A (Detection): Roboflow vehicles dataset
- Part B (Classification): Converted from Roboflow rock-paper-scissors (crops per class)
- Part C (Pose): COCO-pose mini subset (assignment allows "Roboflow / COCO format")
- Part D (OBB): Roboflow dataset converted to OBB label format (4-corner)
"""

import os
import shutil
import yaml
import json
import random
from PIL import Image
from roboflow import Roboflow

API_KEY = "wyukYINpo1eQWmapyHXo"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

rf = Roboflow(api_key=API_KEY)


# ── Part A: Object Detection ─────────────────────────────────────────────────
def download_detection_dataset():
    """Download Roboflow vehicles detection dataset in YOLOv8 format."""
    dest = os.path.join(DATASET_DIR, "detection")
    if os.path.isdir(dest) and os.path.exists(os.path.join(dest, "data.yaml")):
        print("\n[Part A] Detection dataset already exists, skipping.")
        return dest

    print("\n[Part A] Downloading Object Detection dataset...")
    project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
    dataset = project.version(2).download("yolov8", location=dest)
    print(f"  Saved to: {dataset.location}")
    return dataset.location


# ── Part B: Classification ────────────────────────────────────────────────────
def download_classification_dataset():
    """
    Download Roboflow rock-paper-scissors detection dataset, then crop
    each bounding box into per-class folders → train/valid/test/class_name/*.jpg
    This is the folder structure YOLOv8-cls expects.
    """
    cls_base = os.path.join(DATASET_DIR, "classification")
    if os.path.isdir(cls_base) and os.path.isdir(os.path.join(cls_base, "train")):
        print("\n[Part B] Classification dataset already exists, skipping.")
        return cls_base

    print("\n[Part B] Converting detection dataset to classification format...")

    # Reuse the already-downloaded detection dataset (vehicles)
    det_src = os.path.join(DATASET_DIR, "detection")
    if not os.path.isdir(det_src):
        print("  ERROR: Run Part A first (detection dataset needed as source)")
        return None

    with open(os.path.join(det_src, "data.yaml"), "r") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(det_src, split, "images")
        lbl_dir = os.path.join(det_src, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")
            if not os.path.exists(lbl_path):
                continue
            img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
            w, h = img.size
            with open(lbl_path, "r") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = max(0, int((cx - bw / 2) * w))
                    y1 = max(0, int((cy - bh / 2) * h))
                    x2 = min(w, int((cx + bw / 2) * w))
                    y2 = min(h, int((cy + bh / 2) * h))
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    crop = img.crop((x1, y1, x2, y2))
                    cname = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    out_dir = os.path.join(cls_base, split, cname)
                    os.makedirs(out_dir, exist_ok=True)
                    crop.save(os.path.join(out_dir, f"{os.path.splitext(img_file)[0]}_{idx}.jpg"))

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(cls_base, split)
        if os.path.isdir(split_dir):
            classes = os.listdir(split_dir)
            total = sum(len(os.listdir(os.path.join(split_dir, c))) for c in classes)
            print(f"  {split}: {total} images across {len(classes)} classes")

    print(f"  Saved to: {cls_base}")
    return cls_base


# ── Part C: Pose Estimation (COCO keypoints mini subset) ─────────────────────
def download_pose_dataset():
    """
    Download COCO val2017 person-keypoint annotations and a subset of images.
    Creates a YOLOv8-pose compatible dataset with COCO keypoints (17 keypoints).
    Assignment allows 'Roboflow / COCO format' for pose.
    """
    pose_base = os.path.join(DATASET_DIR, "pose")
    if os.path.isdir(pose_base) and os.path.exists(os.path.join(pose_base, "data.yaml")):
        print("\n[Part C] Pose dataset already exists, skipping.")
        return pose_base

    print("\n[Part C] Downloading COCO Pose dataset subset...")
    import urllib.request
    import zipfile

    os.makedirs(pose_base, exist_ok=True)

    # Download COCO person_keypoints val2017 annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = os.path.join(pose_base, "annotations.zip")
    if not os.path.exists(ann_zip):
        print("  Downloading COCO annotations (252 MB)...")
        urllib.request.urlretrieve(ann_url, ann_zip)

    # Extract only person_keypoints_train2017.json and person_keypoints_val2017.json
    print("  Extracting annotations...")
    with zipfile.ZipFile(ann_zip, "r") as z:
        for name in z.namelist():
            if "person_keypoints" in name:
                z.extract(name, pose_base)

    # Parse annotations and pick a subset of images with good keypoints
    train_ann_path = os.path.join(pose_base, "annotations", "person_keypoints_train2017.json")
    val_ann_path = os.path.join(pose_base, "annotations", "person_keypoints_val2017.json")

    def process_coco_split(ann_path, split_name, max_images=200):
        with open(ann_path, "r") as f:
            coco = json.load(f)

        id_to_img = {img["id"]: img for img in coco["images"]}
        # Group annotations by image, filter for good keypoints
        img_anns = {}
        for ann in coco["annotations"]:
            if ann.get("num_keypoints", 0) >= 5 and ann.get("iscrowd", 0) == 0:
                img_id = ann["image_id"]
                if img_id not in img_anns:
                    img_anns[img_id] = []
                img_anns[img_id].append(ann)

        selected_ids = list(img_anns.keys())
        random.seed(42)
        random.shuffle(selected_ids)
        selected_ids = selected_ids[:max_images]

        img_dir = os.path.join(pose_base, split_name, "images")
        lbl_dir = os.path.join(pose_base, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        downloaded = 0
        for img_id in selected_ids:
            info = id_to_img[img_id]
            fname = info["file_name"]
            w, h = info["width"], info["height"]

            # Determine COCO image URL
            coco_split = "train2017" if "train" in ann_path else "val2017"
            url = f"http://images.cocodataset.org/{coco_split}/{fname}"
            dst = os.path.join(img_dir, fname)
            if not os.path.exists(dst):
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception:
                    continue

            # Convert annotations to YOLOv8-pose format:
            # class_id cx cy bw bh kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
            lines = []
            for ann in img_anns[img_id]:
                bx, by, bw_abs, bh_abs = ann["bbox"]  # COCO: x,y,w,h
                cx = (bx + bw_abs / 2) / w
                cy = (by + bh_abs / 2) / h
                nw = bw_abs / w
                nh = bh_abs / h

                kps = ann["keypoints"]  # [x1,y1,v1, x2,y2,v2, ...]
                kp_parts = []
                for k in range(0, len(kps), 3):
                    kx = kps[k] / w
                    ky = kps[k + 1] / h
                    kv = kps[k + 2]  # 0=not labeled, 1=labeled not visible, 2=visible
                    kp_parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

                line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_parts)
                lines.append(line)

            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))
            downloaded += 1

        print(f"  {split_name}: {downloaded} images downloaded")
        return downloaded

    process_coco_split(train_ann_path, "train", max_images=300)
    process_coco_split(val_ann_path, "valid", max_images=80)
    process_coco_split(val_ann_path, "test", max_images=40)

    # Create data.yaml
    kpt_shape = [17, 3]  # COCO: 17 keypoints, each with x,y,visibility
    data_yaml = {
        "path": pose_base,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "person"},
        "kpt_shape": kpt_shape,
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    with open(os.path.join(pose_base, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Clean up large annotation zip
    os.remove(ann_zip)
    shutil.rmtree(os.path.join(pose_base, "annotations"), ignore_errors=True)

    print(f"  Saved to: {pose_base}")
    return pose_base


# ── Part D: OBB (Oriented Bounding Box) ──────────────────────────────────────
def download_obb_dataset():
    """
    Download a Roboflow detection dataset and convert labels to OBB format.
    OBB labels use 4 corner points: class_id x1 y1 x2 y2 x3 y3 x4 y4
    (all normalized). We convert axis-aligned boxes to their 4 corners.
    """
    obb_base = os.path.join(DATASET_DIR, "obb")
    if os.path.isdir(obb_base) and os.path.exists(os.path.join(obb_base, "data.yaml")):
        print("\n[Part D] OBB dataset already exists, skipping.")
        return obb_base

    print("\n[Part D] Downloading & converting OBB dataset...")

    # Use the already-downloaded detection dataset as source
    det_src = os.path.join(DATASET_DIR, "detection")
    if not os.path.isdir(det_src):
        print("  ERROR: Run Part A first (detection dataset needed as source)")
        return None

    with open(os.path.join(det_src, "data.yaml"), "r") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]

    os.makedirs(obb_base, exist_ok=True)

    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(det_src, split, "images")
        src_lbl_dir = os.path.join(det_src, split, "labels")
        if not os.path.isdir(src_img_dir):
            continue

        dst_img_dir = os.path.join(obb_base, split, "images")
        dst_lbl_dir = os.path.join(obb_base, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        count = 0
        for img_file in os.listdir(src_img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            # Symlink/copy image
            src_img = os.path.join(src_img_dir, img_file)
            dst_img = os.path.join(dst_img_dir, img_file)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            # Convert label: YOLO cx cy w h → OBB x1 y1 x2 y2 x3 y3 x4 y4
            lbl_name = os.path.splitext(img_file)[0] + ".txt"
            src_lbl = os.path.join(src_lbl_dir, lbl_name)
            dst_lbl = os.path.join(dst_lbl_dir, lbl_name)

            if not os.path.exists(src_lbl):
                continue

            obb_lines = []
            with open(src_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # 4 corners of axis-aligned box (normalized)
                    x1 = cx - w / 2  # top-left
                    y1 = cy - h / 2
                    x2 = cx + w / 2  # top-right
                    y2 = cy - h / 2
                    x3 = cx + w / 2  # bottom-right
                    y3 = cy + h / 2
                    x4 = cx - w / 2  # bottom-left
                    y4 = cy + h / 2

                    obb_lines.append(
                        f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                        f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                    )

            with open(dst_lbl, "w") as f:
                f.write("\n".join(obb_lines))
            count += 1

        print(f"  {split}: {count} images converted to OBB format")

    # Create data.yaml for OBB
    obb_yaml = {
        "path": obb_base,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_names,
    }
    with open(os.path.join(obb_base, "data.yaml"), "w") as f:
        yaml.dump(obb_yaml, f, default_flow_style=False)

    print(f"  Saved to: {obb_base}")
    return obb_base


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_detection_dataset()
    download_classification_dataset()
    download_pose_dataset()
    download_obb_dataset()

    print("\n" + "=" * 60)
    print("All datasets downloaded successfully!")
    print("=" * 60)
