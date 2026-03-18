"""
test_clipseg.py
================
Evaluation script for the fine-tuned CLIPSeg model.

Outputs
-------
  results/masks/          - prediction PNGs  (e.g. 123__segment_crack.png)
  results/visuals/        - side-by-side orig | GT | pred images
  results/metrics.csv     - per-image IoU & Dice
  results/summary.txt     - aggregate mIoU, mDice, runtime, model size

Usage
-----
  python test_clipseg.py
"""

import os
import time
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

MODEL_WEIGHTS = r"D:\origin\clipseg_drywall_model.pth"

CRACK_IMAGE_DIR  = r"D:\origin\cracks.v1i.yolov8\test\images"
CRACK_MASK_DIR   = r"D:\origin\cracks.v1i.yolov8\test\masks"
CRACK_PROMPT     = "segment crack"

DRYWALL_IMAGE_DIR = r"D:\origin\Drywall_dataset_fixed\test\images"
DRYWALL_MASK_DIR  = r"D:\origin\Drywall_dataset_fixed\test\masks"
DRYWALL_PROMPT    = "segment taping area"

OUTPUT_DIR  = r"D:\origin\results"
INFER_SIZE  = (352, 352)
THRESHOLD   = 0.5
MAX_VISUALS = 4
SEED        = 42

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),   exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visuals"), exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------

print("Loading model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

if os.path.exists(MODEL_WEIGHTS):
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded fine-tuned weights from: {MODEL_WEIGHTS}")
else:
    print(f"WARNING: weights not found at {MODEL_WEIGHTS}. Using base model.")

model.to(device)
model.eval()

model_size_mb = os.path.getsize(MODEL_WEIGHTS) / 1e6 if os.path.exists(MODEL_WEIGHTS) else 0
print(f"Model size on disk: {model_size_mb:.1f} MB")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

to_tensor = T.ToTensor()
resize    = T.Resize(INFER_SIZE)
to_pil    = T.ToPILImage()


def find_mask_path(mask_dir: str, img_stem: str):
    """
    Robustly find a mask file given the image stem.
    Tries exact stem match first, then strips common suffixes like
    _jpg, _png that Roboflow sometimes appends.
    """
    candidates = [
        os.path.join(mask_dir, img_stem + ".png"),
        os.path.join(mask_dir, img_stem + ".jpg"),
    ]
    # also try stripping Roboflow suffixes e.g. "img_jpg.rf.xxx" -> "img"
    clean = img_stem
    for suffix in ("_jpg", "_png", "_jpeg"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            candidates.append(os.path.join(mask_dir, clean + ".png"))
            candidates.append(os.path.join(mask_dir, clean + ".jpg"))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_image_and_mask(img_path: str, mask_path: str):
    """Return (PIL image original size, resized tensor, binary mask tensor [H,W])."""
    img_orig = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_orig.size

    img_t    = to_tensor(resize(img_orig))           # [3,352,352]
    mask_img = Image.open(mask_path).convert("L")
    mask_t   = to_tensor(resize(mask_img))           # [1,352,352]
    mask_bin = (mask_t > 0).float().squeeze(0)       # [352,352]

    return img_orig, img_t, mask_bin, (orig_h, orig_w)


def predict(img_tensor: torch.Tensor, prompt: str, orig_size: tuple):
    """
    Run inference for a single image.
    Returns pred_bin [H,W] at INFER_SIZE, pred_orig np array at orig_size, infer_ms.
    """
    pil_img = to_pil(img_tensor.cpu())

    t0 = time.perf_counter()
    with torch.no_grad():
        inputs  = processor(text=[prompt], images=[pil_img],
                            return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits  = outputs.logits                      # [1,H',W'] or [1,1,H',W']

        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if logits.dim() == 3:
            logits = logits.squeeze(0)                # [H',W']

        logits = F.interpolate(
            logits.unsqueeze(0).unsqueeze(0),
            size=INFER_SIZE,
            mode="bilinear",
            align_corners=False
        ).squeeze()                                   # [352,352]

        prob     = torch.sigmoid(logits)
        pred_bin = (prob >= THRESHOLD).float()        # [352,352]

        pred_orig = F.interpolate(
            pred_bin.unsqueeze(0).unsqueeze(0),
            size=orig_size,
            mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8) * 255   # {0,255}

    infer_ms = (time.perf_counter() - t0) * 1000
    return pred_bin.cpu(), pred_orig, infer_ms


def iou_dice(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.bool()
    gt   = gt.bool()
    intersection = (pred & gt).float().sum()
    union        = (pred | gt).float().sum()
    iou  = (intersection / (union + 1e-6)).item()
    dice = (2 * intersection / (pred.float().sum() + gt.float().sum() + 1e-6)).item()
    return iou, dice


def save_mask(pred_arr: np.ndarray, img_stem: str, prompt: str):
    fname = f"{img_stem}__{prompt.replace(' ', '_')}.png"
    path  = os.path.join(OUTPUT_DIR, "masks", fname)
    Image.fromarray(pred_arr).save(path)
    return fname


def save_visual(img_orig, gt_bin, pred_bin, filename):
    W, H = 352, 352
    canvas = Image.new("RGB", (W * 3 + 20, H + 40), color=(30, 30, 30))

    # original
    canvas.paste(img_orig.resize((W, H)), (0, 40))

    def overlay(base_img, mask_np, color):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, color] = mask_np
        base_rgba = base_img.resize((W, H)).convert("RGBA")
        ov = Image.fromarray(rgb).convert("RGBA")
        ov.putalpha(Image.fromarray((mask_np * 180).astype(np.uint8)))
        base_rgba.paste(ov, (0, 0), ov)
        return base_rgba.convert("RGB")

    gt_np   = (gt_bin.numpy()   * 255).astype(np.uint8)
    pred_np = (pred_bin.numpy() * 255).astype(np.uint8)

    canvas.paste(overlay(img_orig, gt_np,   1), (W + 10,      40))   # green
    canvas.paste(overlay(img_orig, pred_np, 0), (W * 2 + 20,  40))   # red

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((W // 2 - 30,              8), "Original",     fill="white", font=font)
    draw.text((W + 10 + W // 2 - 45,    8), "Ground Truth", fill="lime",  font=font)
    draw.text((W * 2 + 20 + W // 2 - 40, 8), "Prediction",  fill="red",   font=font)

    canvas.save(os.path.join(OUTPUT_DIR, "visuals", filename))
    print(f"  Visual saved: {filename}")


# -----------------------------------------------------------------------------
# EVALUATE ONE DATASET
# -----------------------------------------------------------------------------

def evaluate_dataset(image_dir, mask_dir, prompt, label):
    print(f"\n{'='*60}")
    print(f"Evaluating : {label}")
    print(f"Prompt     : '{prompt}'")
    print(f"Images     : {image_dir}")
    print(f"Masks      : {mask_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(image_dir):
        print(f"  ERROR: image directory not found:\n  {image_dir}")
        return [], None

    if not os.path.isdir(mask_dir):
        print(f"  ERROR: mask directory not found:\n  {mask_dir}")
        return [], None

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"  No images found in {image_dir}")
        return [], None

    print(f"  Found {len(image_files)} images")

    iou_list, dice_list, time_list = [], [], []
    rows = []
    skipped = 0

    visual_indices = set(random.sample(range(len(image_files)),
                                       min(MAX_VISUALS, len(image_files))))

    for i, img_name in enumerate(image_files):
        img_stem  = Path(img_name).stem
        img_path  = os.path.join(image_dir, img_name)
        mask_path = find_mask_path(mask_dir, img_stem)

        if mask_path is None:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipping {img_name} — no matching mask found")
            continue

        try:
            img_orig, img_t, gt_bin, orig_size = load_image_and_mask(img_path, mask_path)
            pred_bin, pred_orig, ms            = predict(img_t, prompt, orig_size)

            iou, dice = iou_dice(pred_bin, gt_bin)
            iou_list.append(iou)
            dice_list.append(dice)
            time_list.append(ms)

            save_mask(pred_orig, img_stem, prompt)

            rows.append({
                "dataset":  label,
                "image":    img_name,
                "prompt":   prompt,
                "iou":      round(iou, 4),
                "dice":     round(dice, 4),
                "infer_ms": round(ms, 1)
            })

            if i in visual_indices:
                vis_name = f"{label.replace(' ', '_')}_{img_stem}_visual.png"
                save_visual(img_orig, gt_bin, pred_bin, vis_name)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_files)}]  "
                      f"IoU={iou:.3f}  Dice={dice:.3f}  {ms:.0f}ms")

        except Exception as e:
            print(f"  ERROR on {img_name}: {e}")
            skipped += 1
            continue

    if skipped:
        print(f"  Total skipped: {skipped}")

    if not iou_list:
        print("  No images evaluated successfully.")
        return rows, None

    m_iou  = float(np.mean(iou_list))
    m_dice = float(np.mean(dice_list))
    m_time = float(np.mean(time_list))

    print(f"\n  Results for '{label}':")
    print(f"    Images evaluated : {len(iou_list)}")
    print(f"    mIoU             : {m_iou:.4f}")
    print(f"    mDice            : {m_dice:.4f}")
    print(f"    Avg infer time   : {m_time:.1f} ms/image")

    return rows, {
        "label":  label,
        "prompt": prompt,
        "n":      len(iou_list),
        "mIoU":   m_iou,
        "mDice":  m_dice,
        "avg_ms": m_time
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

all_rows    = []
all_summary = []

t_start = time.perf_counter()

crack_rows,   crack_summary   = evaluate_dataset(
    CRACK_IMAGE_DIR, CRACK_MASK_DIR, CRACK_PROMPT, "Cracks")

drywall_rows, drywall_summary = evaluate_dataset(
    DRYWALL_IMAGE_DIR, DRYWALL_MASK_DIR, DRYWALL_PROMPT, "Drywall Taping")

total_time = time.perf_counter() - t_start

all_rows    = crack_rows + drywall_rows
all_summary = [s for s in [crack_summary, drywall_summary] if s is not None]

# ── metrics.csv ──────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["dataset", "image", "prompt", "iou", "dice", "infer_ms"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nPer-image metrics : {csv_path}")

# ── summary.txt ───────────────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  CLIPSeg Drywall QA — Evaluation Summary\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Model weights  : {MODEL_WEIGHTS}\n")
    f.write(f"Model size     : {model_size_mb:.1f} MB\n")
    f.write(f"Device         : {device}\n")
    f.write(f"Threshold      : {THRESHOLD}\n")
    f.write(f"Infer size     : {INFER_SIZE}\n")
    f.write(f"Seed           : {SEED}\n")
    f.write(f"Total eval time: {total_time:.1f}s\n\n")

    f.write(f"{'Dataset':<22} {'Prompt':<26} {'N':>5} "
            f"{'mIoU':>8} {'mDice':>8} {'ms/img':>8}\n")
    f.write("-" * 80 + "\n")
    for s in all_summary:
        f.write(f"{s['label']:<22} {s['prompt']:<26} {s['n']:>5} "
                f"{s['mIoU']:>8.4f} {s['mDice']:>8.4f} {s['avg_ms']:>8.1f}\n")

    if all_rows:
        all_iou  = [r["iou"]  for r in all_rows]
        all_dice = [r["dice"] for r in all_rows]
        f.write("-" * 80 + "\n")
        f.write(f"{'OVERALL':<22} {'':<26} {len(all_iou):>5} "
                f"{np.mean(all_iou):>8.4f} {np.mean(all_dice):>8.4f}\n")

print(f"Summary           : {summary_path}")

# ── console table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULTS")
print("=" * 55)
print(f"{'Dataset':<22} {'mIoU':>8} {'mDice':>8} {'ms/img':>8}")
print("-" * 50)
for s in all_summary:
    print(f"{s['label']:<22} {s['mIoU']:>8.4f} {s['mDice']:>8.4f} {s['avg_ms']:>8.1f}")
if all_rows:
    all_iou  = [r["iou"]  for r in all_rows]
    all_dice = [r["dice"] for r in all_rows]
    print("-" * 50)
    print(f"{'OVERALL':<22} {np.mean(all_iou):>8.4f} {np.mean(all_dice):>8.4f}")

print(f"\nAll outputs saved in: {OUTPUT_DIR}")
print("Done.")