import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SOURCE_BASE  = r"D:\origin\cracks.v1i.yolov8"   # original dataset
OUTPUT_BASE  = r"D:\origin\cracks_split"          # new balanced dataset

SOURCE_SPLITS = ["train", "test", "valid"]        # folders to pool from
SUBFOLDERS    = ["images", "labels", "masks", "bbox_images"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def find_file(folder, stem):
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        if Path(f).stem == stem:
            return os.path.join(folder, f)
    return None

def copy_sample(stem, src_split_dir, dst_split_dir):
    copied = []
    for sub in SUBFOLDERS:
        src_sub  = os.path.join(src_split_dir, sub)
        dst_sub  = os.path.join(dst_split_dir, sub)
        src_file = find_file(src_sub, stem)
        if src_file:
            os.makedirs(dst_sub, exist_ok=True)
            shutil.copy2(src_file, os.path.join(dst_sub, os.path.basename(src_file)))
            copied.append(sub)
    return copied

# ─────────────────────────────────────────────────────────────────
# STEP 1 — POOL ALL IMAGES
# ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Crack Dataset Re-Splitter")
print("=" * 55)
print(f"Source : {SOURCE_BASE}")
print(f"Output : {OUTPUT_BASE}")
print()

all_samples = {}  # stem -> split_dir

for split in SOURCE_SPLITS:
    split_dir  = os.path.join(SOURCE_BASE, split)
    images_dir = os.path.join(split_dir, "images")

    if not os.path.isdir(images_dir):
        print(f"  [{split}] images/ not found — skipping")
        continue

    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"  [{split}] {len(files)} images found")

    for f in files:
        stem = Path(f).stem
        if stem not in all_samples:
            all_samples[stem] = split_dir

total = len(all_samples)
print(f"\nTotal unique images: {total}")

if total == 0:
    print("ERROR: No images found — check SOURCE_BASE path")
    exit(1)

# ─────────────────────────────────────────────────────────────────
# STEP 2 — SHUFFLE AND SPLIT
# ─────────────────────────────────────────────────────────────────

random.seed(SEED)
stems = list(all_samples.keys())
random.shuffle(stems)

n_train = int(total * TRAIN_RATIO)
n_val   = int(total * VAL_RATIO)
n_test  = total - n_train - n_val

train_stems = stems[:n_train]
val_stems   = stems[n_train : n_train + n_val]
test_stems  = stems[n_train + n_val:]

print(f"\nSplit (seed={SEED}, ratio 70/15/15):")
print(f"  train : {len(train_stems)}")
print(f"  val   : {len(val_stems)}")
print(f"  test  : {len(test_stems)}")
print(f"  total : {total}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — COPY FILES
# ─────────────────────────────────────────────────────────────────

print(f"\nCopying files... (may take a few minutes)")

for split_name, split_stems in [("train", train_stems),
                                  ("val",   val_stems),
                                  ("test",  test_stems)]:
    dst_dir = os.path.join(OUTPUT_BASE, split_name)
    ok = 0
    missing = 0
    for stem in split_stems:
        result = copy_sample(stem, all_samples[stem], dst_dir)
        if "images" in result:
            ok += 1
        else:
            missing += 1
    print(f"  [{split_name}] copied={ok}  missing={missing}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — VERIFY
# ─────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  FINAL COUNTS")
print(f"{'='*55}")

for split_name in ["train", "val", "test"]:
    print(f"\n  [{split_name}]")
    for sub in SUBFOLDERS:
        sub_dir = os.path.join(OUTPUT_BASE, split_name, sub)
        if os.path.isdir(sub_dir):
            print(f"    {sub:<15}: {len(os.listdir(sub_dir))}")
        else:
            print(f"    {sub:<15}: not created")

print(f"\nDone! New dataset at: {OUTPUT_BASE}")
print("\nNext — update your train/test scripts:")
print(f"  CRACK_IMAGE_DIR = r'{OUTPUT_BASE}\\train\\images'")
print(f"  CRACK_MASK_DIR  = r'{OUTPUT_BASE}\\train\\masks'")