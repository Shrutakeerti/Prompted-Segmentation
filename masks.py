"""
generate_masks.py
==================
Generates binary segmentation masks from YOLO bounding-box label files.
Drywall dataset only — processes train, test, val splits.

For each image:
  - Reads YOLO .txt label (class cx cy w h — normalized 0-1)
  - Draws all bboxes as filled white rectangles on black canvas
  - Saves as single-channel PNG mask {0, 255}

Folder structure:
  Drywall_dataset_fixed/
    train/
      images/   ← source images
      labels/   ← YOLO .txt files
      masks/    ← OUTPUT (overwritten)
    test/   (same)
    val/    (same)

Usage:
  python generate_masks.py
"""

import os
from PIL import Image, ImageDraw
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_DIR = r"D:\origin\Drywall_dataset_fixed"
SPLITS   = ["train", "test", "val"]

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def make_mask(label_path, img_w, img_h):
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    boxes = 0
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, bw, bh = (float(p) for p in parts[:5])
            except ValueError:
                continue
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            boxes += 1
    return mask, boxes


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

print(f"Dataset: {BASE_DIR}\n")

total_written = 0

for split in SPLITS:
    images_dir = os.path.join(BASE_DIR, split, "images")
    labels_dir = os.path.join(BASE_DIR, split, "labels")
    masks_dir  = os.path.join(BASE_DIR, split, "masks")

    if not os.path.isdir(images_dir):
        print(f"[{split}] SKIP — images not found: {images_dir}")
        continue
    if not os.path.isdir(labels_dir):
        print(f"[{split}] SKIP — labels not found: {labels_dir}")
        continue

    os.makedirs(masks_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    print(f"[{split}]  {len(label_files)} labels  →  {masks_dir}")

    written = no_img = no_box = errors = 0

    for lbl_file in label_files:
        stem       = Path(lbl_file).stem
        label_path = os.path.join(labels_dir, lbl_file)
        img_path   = find_image(images_dir, stem)

        if img_path is None:
            no_img += 1
            if no_img <= 3:
                print(f"  WARNING: no image for '{stem}'")
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            mask, n_boxes = make_mask(label_path, img_w, img_h)

            if n_boxes == 0:
                no_box += 1  # saves blank mask — background only image

            mask.save(os.path.join(masks_dir, stem + ".png"))
            written += 1

        except Exception as e:
            print(f"  ERROR {lbl_file}: {e}")
            errors += 1

    print(f"  Written   : {written}")
    print(f"  No image  : {no_img}")
    print(f"  No boxes  : {no_box}  (blank masks)")
    print(f"  Errors    : {errors}\n")
    total_written += written

print(f"Total masks written: {total_written}")
print("\nDone. Now retrain with train_drywall.py")