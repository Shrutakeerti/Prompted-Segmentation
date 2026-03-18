"""
test_drywall.py  —  Fixed version with auto-threshold + diagnostics
"""

import os
import time
import csv
import random
import base64
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

MODEL_WEIGHTS = r"D:\origin\clipseg_drywall.pth"

IMAGE_DIR = r"D:\origin\Drywall_dataset_fixed\test\images"
MASK_DIR  = r"D:\origin\Drywall_dataset_fixed\test\masks"
PROMPT    = "segment taping area"

OUTPUT_DIR  = r"D:\origin\results_drywall"
INFER_SIZE  = (352, 352)

# THRESHOLD: set to None for AUTO (uses Otsu on each image)
# or set a float e.g. 0.3, 0.2 to use fixed threshold
THRESHOLD   = None

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
    print(f"Loaded fine-tuned weights: {MODEL_WEIGHTS}")
else:
    print(f"WARNING: weights not found — using base model")

model.to(device)
model.eval()

model_size_mb = os.path.getsize(MODEL_WEIGHTS) / 1e6 if os.path.exists(MODEL_WEIGHTS) else 0
print(f"Model size : {model_size_mb:.1f} MB")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

resize = T.Resize(INFER_SIZE)


def find_mask_path(mask_dir, img_stem):
    candidates = [
        os.path.join(mask_dir, img_stem + ".png"),
        os.path.join(mask_dir, img_stem + ".jpg"),
    ]
    clean = img_stem
    for suffix in ("_jpg", "_png", "_jpeg"):
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
            candidates += [os.path.join(mask_dir, clean + ".png"),
                           os.path.join(mask_dir, clean + ".jpg")]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def otsu_threshold(prob_np):
    """Compute Otsu threshold on a float [0,1] probability map."""
    img_u8 = (prob_np * 255).astype(np.uint8)
    counts, _ = np.histogram(img_u8, bins=256, range=(0, 255))
    total = img_u8.size
    sum_all = np.dot(np.arange(256), counts)
    sum_bg, w_bg, best_thresh, best_var = 0.0, 0, 0, 0.0
    for t in range(256):
        w_bg += counts[t]
        if w_bg == 0: continue
        w_fg = total - w_bg
        if w_fg == 0: break
        sum_bg += t * counts[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t
    return best_thresh / 255.0


def load_image_and_mask(img_path, mask_path):
    img_orig       = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_orig.size
    img_resized    = resize(img_orig)                       # PIL, 352x352
    mask_t         = T.ToTensor()(resize(
                         Image.open(mask_path).convert("L")))
    mask_bin       = (mask_t > 0).float().squeeze(0)       # [352,352]
    return img_orig, img_resized, mask_bin, (orig_h, orig_w)


def predict(pil_img_352, prompt, orig_size):
    """
    pil_img_352: PIL image already resized to 352x352
    Returns pred_bin [352,352], pred_orig np array, infer_ms, prob_np, used_threshold
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        # Pass PIL image directly — no tensor conversion
        inputs = processor(
            text=[prompt],
            images=[pil_img_352],
            return_tensors="pt",
            padding=True
        ).to(device)

        logits = model(**inputs).logits       # [1,H',W'] or [1,1,H',W']

        if logits.dim() == 4: logits = logits.squeeze(1)
        if logits.dim() == 3: logits = logits.squeeze(0)  # [H',W']

        logits = F.interpolate(
            logits.unsqueeze(0).unsqueeze(0),
            size=INFER_SIZE, mode="bilinear", align_corners=False
        ).squeeze()                           # [352,352]

        prob_np = torch.sigmoid(logits).cpu().numpy()      # float [0,1]

    # --- choose threshold ---
    if THRESHOLD is None:
        used_thresh = otsu_threshold(prob_np)
        # fallback: if otsu gives 0 or 1 (flat image), use 0.3
        if used_thresh < 0.05 or used_thresh > 0.95:
            used_thresh = 0.3
    else:
        used_thresh = THRESHOLD

    pred_bin = torch.from_numpy((prob_np >= used_thresh).astype(np.float32))  # [352,352]

    pred_orig = F.interpolate(
        pred_bin.unsqueeze(0).unsqueeze(0),
        size=orig_size, mode="nearest"
    ).squeeze().numpy().astype(np.uint8) * 255

    infer_ms = (time.perf_counter() - t0) * 1000
    return pred_bin, pred_orig, infer_ms, prob_np, used_thresh


def iou_dice(pred, gt):
    pred = pred.bool(); gt = gt.bool()
    inter = (pred & gt).float().sum()
    union = (pred | gt).float().sum()
    iou   = (inter / (union + 1e-6)).item()
    dice  = (2 * inter / (pred.float().sum() + gt.float().sum() + 1e-6)).item()
    return iou, dice


def save_mask(pred_arr, img_stem, prompt):
    fname = f"{img_stem}__{prompt.replace(' ', '_')}.png"
    Image.fromarray(pred_arr).save(os.path.join(OUTPUT_DIR, "masks", fname))


def save_visual(img_orig, gt_bin, pred_bin, prob_np, thresh, filename):
    W, H = 352, 352
    # 4-panel: Original | GT | Prediction | Probability heatmap
    canvas = Image.new("RGB", (W * 4 + 30, H + 60), (30, 30, 30))

    def overlay(base, mask_np, ch):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, ch] = mask_np
        base_rgba = base.resize((W, H)).convert("RGBA")
        ov = Image.fromarray(rgb).convert("RGBA")
        ov.putalpha(Image.fromarray((mask_np * 180).astype(np.uint8)))
        base_rgba.paste(ov, (0, 0), ov)
        return base_rgba.convert("RGB")

    # probability heatmap (jet colormap manually)
    prob_u8  = (np.clip(prob_np, 0, 1) * 255).astype(np.uint8)
    prob_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # simple blue→green→red ramp
    prob_rgb[:, :, 0] = np.clip(prob_u8 * 2 - 255, 0, 255)          # red
    prob_rgb[:, :, 1] = np.clip(255 - np.abs(prob_u8.astype(int) - 128) * 2, 0, 255)  # green
    prob_rgb[:, :, 2] = np.clip(255 - prob_u8 * 2, 0, 255)           # blue
    heatmap = Image.fromarray(prob_rgb).resize((W, H))

    gt_np   = (gt_bin.numpy()   * 255).astype(np.uint8)
    pred_np = (pred_bin.numpy() * 255).astype(np.uint8)

    canvas.paste(img_orig.resize((W, H)),              (0,           60))
    canvas.paste(overlay(img_orig, gt_np,   1),        (W + 10,      60))
    canvas.paste(overlay(img_orig, pred_np, 0),        (W * 2 + 20,  60))
    canvas.paste(heatmap,                              (W * 3 + 30,  60))

    draw = ImageDraw.Draw(canvas)
    try:    font = ImageFont.truetype("arial.ttf", 14)
    except: font = ImageFont.load_default()

    labels = ["Original", "Ground Truth", f"Pred (t={thresh:.2f})", "Prob Heatmap"]
    colors = ["white",    "lime",         "red",                     "yellow"]
    for j, (lbl, col) in enumerate(zip(labels, colors)):
        x = j * (W + 10) + W // 2 - len(lbl) * 4
        draw.text((x, 8), lbl, fill=col, font=font)

    out_path = os.path.join(OUTPUT_DIR, "visuals", filename)
    canvas.save(out_path)
    print(f"  Visual saved: {filename}  (threshold used: {thresh:.3f})")
    return out_path


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# -----------------------------------------------------------------------------
# MAIN EVALUATION
# -----------------------------------------------------------------------------

print(f"\nPrompt : '{PROMPT}'")
print(f"Images : {IMAGE_DIR}")
print(f"Masks  : {MASK_DIR}")
print(f"Threshold mode: {'AUTO (Otsu per image)' if THRESHOLD is None else THRESHOLD}\n")

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
print(f"Found {len(image_files)} images")

iou_list, dice_list, time_list, thresh_list = [], [], [], []
rows, visual_paths = [], []
skipped = 0

visual_indices = set(random.sample(range(len(image_files)),
                                   min(MAX_VISUALS, len(image_files))))

# --- diagnostic on first 3 images ---
print("\n--- DIAGNOSTIC (first 3 images) ---")
for img_name in image_files[:3]:
    img_stem  = Path(img_name).stem
    mask_path = find_mask_path(MASK_DIR, img_stem)
    if mask_path is None:
        print(f"  {img_name}: no mask found"); continue
    img_orig, img_resized, gt_bin, orig_size = load_image_and_mask(
        os.path.join(IMAGE_DIR, img_name), mask_path)
    with torch.no_grad():
        inputs = processor(text=[PROMPT], images=[img_resized],
                           return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits
        if logits.dim() == 4: logits = logits.squeeze(1)
        if logits.dim() == 3: logits = logits.squeeze(0)
        logits = F.interpolate(logits.unsqueeze(0).unsqueeze(0),
                               size=INFER_SIZE, mode="bilinear",
                               align_corners=False).squeeze()
        prob = torch.sigmoid(logits).cpu().numpy()
    auto_t = otsu_threshold(prob)
    gt_pos = gt_bin.sum().item()
    print(f"  {img_name}")
    print(f"    prob  min={prob.min():.4f}  max={prob.max():.4f}  "
          f"mean={prob.mean():.4f}  >0.5: {(prob>0.5).sum()}")
    print(f"    GT positive pixels: {int(gt_pos)}")
    print(f"    Otsu threshold: {auto_t:.4f}")
print("--- END DIAGNOSTIC ---\n")

t_start = time.perf_counter()

for i, img_name in enumerate(image_files):
    img_stem  = Path(img_name).stem
    img_path  = os.path.join(IMAGE_DIR, img_name)
    mask_path = find_mask_path(MASK_DIR, img_stem)

    if mask_path is None:
        skipped += 1
        if skipped <= 3: print(f"  Skipping {img_name} — no mask")
        continue

    try:
        img_orig, img_resized, gt_bin, orig_size = load_image_and_mask(img_path, mask_path)
        pred_bin, pred_orig, ms, prob_np, thresh = predict(img_resized, PROMPT, orig_size)
        iou, dice = iou_dice(pred_bin, gt_bin)

        iou_list.append(iou);  dice_list.append(dice)
        time_list.append(ms);  thresh_list.append(thresh)

        save_mask(pred_orig, img_stem, PROMPT)

        rows.append({"image": img_name, "prompt": PROMPT,
                     "iou": round(iou,4), "dice": round(dice,4),
                     "threshold": round(thresh,4), "infer_ms": round(ms,1)})

        if i in visual_indices:
            vpath = save_visual(img_orig, gt_bin, pred_bin, prob_np, thresh,
                                f"drywall_{img_stem}_visual.png")
            visual_paths.append(vpath)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [{i+1}/{len(image_files)}]  "
                  f"IoU={iou:.3f}  Dice={dice:.3f}  thresh={thresh:.3f}  {ms:.0f}ms")

    except Exception as e:
        print(f"  ERROR on {img_name}: {e}"); skipped += 1

total_time = time.perf_counter() - t_start

# -----------------------------------------------------------------------------
# SAVE OUTPUTS
# -----------------------------------------------------------------------------

if not iou_list:
    print("No images evaluated. Check your paths."); exit(1)

m_iou  = float(np.mean(iou_list))
m_dice = float(np.mean(dice_list))
m_time = float(np.mean(time_list))
m_thr  = float(np.mean(thresh_list))

# metrics.csv
csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image","prompt","iou","dice","threshold","infer_ms"])
    writer.writeheader(); writer.writerows(rows)

# summary.txt
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write("=" * 60 + "\n  CLIPSeg Drywall — Evaluation Summary\n" + "=" * 60 + "\n\n")
    f.write(f"Model          : {MODEL_WEIGHTS}\n")
    f.write(f"Model size     : {model_size_mb:.1f} MB\n")
    f.write(f"Device         : {device}\n")
    f.write(f"Prompt         : {PROMPT}\n")
    f.write(f"Threshold mode : {'AUTO Otsu' if THRESHOLD is None else THRESHOLD}\n")
    f.write(f"Avg threshold  : {m_thr:.4f}\n")
    f.write(f"Seed           : {SEED}\n")
    f.write(f"Total eval time: {total_time:.1f}s\n\n")
    f.write(f"Images : {len(iou_list)}\n")
    f.write(f"mIoU   : {m_iou:.4f}\n")
    f.write(f"mDice  : {m_dice:.4f}\n")
    f.write(f"ms/img : {m_time:.1f}\n")

# HTML report
def badge(v):
    c = "#2ecc71" if v >= 0.7 else ("#f39c12" if v >= 0.5 else "#e74c3c")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{v:.4f}</span>'

vis_html = ""
for vpath in visual_paths:
    if os.path.exists(vpath):
        b64 = img_to_base64(vpath)
        vis_html += f'''<div style="margin:12px 0">
          <p style="font-size:11px;color:#999">{os.path.basename(vpath)}</p>
          <img src="data:image/png;base64,{b64}"
               style="width:100%;max-width:1200px;border:1px solid #ddd;border-radius:4px"/>
        </div>'''

detail_html = ""
for r in rows[:300]:
    ic = "#2ecc71" if r["iou"]  >= 0.7 else ("#f39c12" if r["iou"]  >= 0.5 else "#e74c3c")
    dc = "#2ecc71" if r["dice"] >= 0.7 else ("#f39c12" if r["dice"] >= 0.5 else "#e74c3c")
    detail_html += f"""<tr>
      <td style="font-size:11px">{r['image']}</td>
      <td><span style="color:{ic};font-weight:bold">{r['iou']:.4f}</span></td>
      <td><span style="color:{dc};font-weight:bold">{r['dice']:.4f}</span></td>
      <td>{r['threshold']}</td>
      <td>{r['infer_ms']} ms</td></tr>"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>CLIPSeg Drywall Report</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:0 auto;padding:30px;color:#2c3e50;background:#fafafa}}
  h1{{color:#1a252f;border-bottom:3px solid #3498db;padding-bottom:10px}}
  h2{{color:#2980b9;margin-top:40px}}
  table{{border-collapse:collapse;width:100%;margin:15px 0;background:white;
         box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:6px;overflow:hidden}}
  th{{background:#2c3e50;color:white;padding:10px 14px;text-align:left;font-size:13px}}
  td{{padding:8px 14px;border-bottom:1px solid #ecf0f1;font-size:13px}}
  tr:hover td{{background:#f7f9fa}}
  .meta{{background:white;border-left:4px solid #3498db;padding:14px 20px;
          margin:20px 0;border-radius:0 6px 6px 0;font-size:13px}}
  .meta p{{margin:4px 0}}
</style></head><body>
<h1>🧱 CLIPSeg Drywall Taping — Evaluation Report</h1>
<div class="meta">
  <p><b>Model :</b> {MODEL_WEIGHTS}</p>
  <p><b>Model size :</b> {model_size_mb:.1f} MB &nbsp;|&nbsp; <b>Device :</b> {device}</p>
  <p><b>Prompt :</b> {PROMPT}</p>
  <p><b>Threshold :</b> {'AUTO (Otsu per image)' if THRESHOLD is None else THRESHOLD} &nbsp;|&nbsp; <b>Avg threshold :</b> {m_thr:.4f}</p>
  <p><b>Seed :</b> {SEED} &nbsp;|&nbsp; <b>Total eval time :</b> {total_time:.1f}s</p>
</div>
<h2>📊 Aggregate Metrics</h2>
<table><thead><tr><th>Images</th><th>mIoU</th><th>mDice</th><th>Avg Threshold</th><th>Avg ms/img</th></tr></thead>
<tbody><tr><td>{len(iou_list)}</td><td>{badge(m_iou)}</td><td>{badge(m_dice)}</td>
<td>{m_thr:.4f}</td><td>{m_time:.1f}</td></tr></tbody></table>
<h2>🖼️ Visual Examples (Original | GT | Prediction | Probability Heatmap)</h2>
{vis_html or '<p style="color:#888">No visuals.</p>'}
<h2>📋 Per-Image Results</h2>
<table><thead><tr><th>Image</th><th>IoU</th><th>Dice</th><th>Threshold</th><th>Infer Time</th></tr></thead>
<tbody>{detail_html}</tbody></table>
</body></html>"""

report_path = os.path.join(OUTPUT_DIR, "report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

# Console
print(f"\n{'='*50}")
print(f"  FINAL RESULTS — Drywall Taping")
print(f"{'='*50}")
print(f"  Images   : {len(iou_list)}")
print(f"  mIoU     : {m_iou:.4f}")
print(f"  mDice    : {m_dice:.4f}")
print(f"  Avg thresh: {m_thr:.4f}")
print(f"  ms/image : {m_time:.1f}")
print(f"{'='*50}")
print(f"\nOutputs in: {OUTPUT_DIR}")
print(f"  report.html  ← open in browser")
print(f"  metrics.csv  ← open in Excel")
print(f"  summary.txt")
print(f"  masks\\  visuals\\")
print("\nDone.")