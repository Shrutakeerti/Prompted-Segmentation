"""
split_and_train_cracks.py
==========================
Step 1: Takes 800 random images from cracks.v1i.yolov8 (all splits pooled)
        and divides them:  train=500  val=200  test=100
        Saves to: D:\\origin\\cracks_small\\

Step 2: Trains CLIPSeg on the 500 train images for 5 epochs
        Saves model to: D:\\origin\\clipseg_cracks.pth

Usage:
  python split_and_train_cracks.py
"""

import os
import random
import shutil
import warnings
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torchvision.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SOURCE_BASE   = r"D:\origin\cracks.v1i.yolov8"
OUTPUT_BASE   = r"D:\origin\cracks_small"
SOURCE_SPLITS = ["train", "test", "valid"]
SUBFOLDERS    = ["images", "labels", "masks", "bbox_images"]

N_TRAIN = 500
N_VAL   = 200
N_TEST  = 100
TOTAL   = N_TRAIN + N_VAL + N_TEST   # 800

SAVE_PATH = r"D:\origin\clipseg_cracks.pth"

PROMPTS = [
    "segment crack",
    "segment wall crack",
    "segment surface crack",
]

BATCH  = 4
EPOCHS = 10
LR     = 5e-6
SEED   = 42

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

def find_mask(mask_dir, stem):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def copy_sample(stem, src_dir, dst_dir):
    copied = []
    for sub in SUBFOLDERS:
        src_file = find_file(os.path.join(src_dir, sub), stem)
        if src_file:
            dst_sub = os.path.join(dst_dir, sub)
            os.makedirs(dst_sub, exist_ok=True)
            shutil.copy2(src_file, os.path.join(dst_sub, os.path.basename(src_file)))
            copied.append(sub)
    return copied

# ═════════════════════════════════════════════════════════════════
# STEP 1 — SPLIT
# ═════════════════════════════════════════════════════════════════

print("=" * 55)
print("  STEP 1 — Creating cracks_small (500/200/100)")
print("=" * 55)

# pool all images
all_samples = {}
for split in SOURCE_SPLITS:
    img_dir = os.path.join(SOURCE_BASE, split, "images")
    if not os.path.isdir(img_dir):
        print(f"  [{split}] not found — skipping")
        continue
    files = [f for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  [{split}] {len(files)} images")
    for f in files:
        stem = Path(f).stem
        if stem not in all_samples:
            all_samples[stem] = os.path.join(SOURCE_BASE, split)

total_available = len(all_samples)
print(f"\n  Total available: {total_available}")

if total_available < TOTAL:
    print(f"  WARNING: only {total_available} images available, "
          f"using all instead of {TOTAL}")
    TOTAL   = total_available
    N_TRAIN = int(TOTAL * 0.625)
    N_VAL   = int(TOTAL * 0.25)
    N_TEST  = TOTAL - N_TRAIN - N_VAL

# shuffle and pick TOTAL images
random.seed(SEED)
stems = list(all_samples.keys())
random.shuffle(stems)
selected = stems[:TOTAL]

train_stems = selected[:N_TRAIN]
val_stems   = selected[N_TRAIN : N_TRAIN + N_VAL]
test_stems  = selected[N_TRAIN + N_VAL:]

print(f"\n  Split (seed={SEED}):")
print(f"    train : {len(train_stems)}")
print(f"    val   : {len(val_stems)}")
print(f"    test  : {len(test_stems)}")

# copy files
print(f"\n  Copying to {OUTPUT_BASE} ...")
for split_name, split_stems in [("train", train_stems),
                                  ("val",   val_stems),
                                  ("test",  test_stems)]:
    dst = os.path.join(OUTPUT_BASE, split_name)
    ok = miss = 0
    for stem in split_stems:
        r = copy_sample(stem, all_samples[stem], dst)
        if "images" in r: ok += 1
        else: miss += 1
    print(f"    [{split_name}] copied={ok}  missing={miss}")

# verify
print(f"\n  Verification:")
for spl in ["train", "val", "test"]:
    img_dir = os.path.join(OUTPUT_BASE, spl, "images")
    msk_dir = os.path.join(OUTPUT_BASE, spl, "masks")
    n_img = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    n_msk = len(os.listdir(msk_dir)) if os.path.isdir(msk_dir) else 0
    print(f"    [{spl}]  images={n_img}  masks={n_msk}")

print("\n  Split complete!")

# ═════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN
# ═════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  STEP 2 — Training CLIPSeg on 500 crack images")
print("=" * 55)

IMAGE_DIR = os.path.join(OUTPUT_BASE, "train", "images")
MASK_DIR  = os.path.join(OUTPUT_BASE, "train", "masks")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice  : {device}")
print(f"Images  : {IMAGE_DIR}")
print(f"Save to : {SAVE_PATH}")

# ── MODEL ────────────────────────────────────────────────────────
print("\nLoading model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)
print("Model loaded.")

# ── DATASET ──────────────────────────────────────────────────────

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.mask_tf  = T.Compose([T.Resize((352, 352)), T.ToTensor()])
        all_imgs = [f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.images = [f for f in all_imgs
                       if find_mask(mask_dir, Path(f).stem)]
        skipped = len(all_imgs) - len(self.images)
        if skipped: print(f"  Skipped {skipped} images (no mask)")
        print(f"  Dataset ready: {len(self.images)} images")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        name  = self.images[idx]
        stem  = Path(name).stem
        image = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask  = Image.open(find_mask(self.mask_dir, stem)).convert("L")
        mask  = self.mask_tf(mask)
        mask  = (mask > 0).float().squeeze(0)   # [352, 352]
        return image, mask, random.choice(PROMPTS)

def collate_fn(batch):
    return ([b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch])

dataset = CrackDataset(IMAGE_DIR, MASK_DIR)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                     num_workers=0, collate_fn=collate_fn)

print(f"\nTraining samples : {len(dataset)}")
print(f"Batches/epoch    : {len(loader)}")
print(f"Epochs           : {EPOCHS}  |  LR : {LR}  |  Batch : {BATCH}")
est = len(loader) * EPOCHS * 3 // 60
print(f"Est. time on CPU : ~{est} min\n")

# ── TRAIN LOOP ───────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, masks, prompts in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        masks = masks.to(device)

        text_inp = processor.tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        img_inp = processor.image_processor(
            images=images, return_tensors="pt"
        ).to(device)

        outputs = model(**{**text_inp, **img_inp})
        preds   = outputs.logits

        if preds.dim() == 4: preds = preds.squeeze(1)
        preds = F.interpolate(
            preds.unsqueeze(1), size=(352, 352),
            mode="bilinear", align_corners=False
        ).squeeze(1)                        # [B, 352, 352]

        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  —  Avg Loss: {avg:.4f}")

    # sanity check
    model.eval()
    with torch.no_grad():
        img0, _, p0 = dataset[0]
        t = processor.tokenizer([p0], return_tensors="pt",
                                  padding=True, truncation=True).to(device)
        i = processor.image_processor(images=[img0],
                                       return_tensors="pt").to(device)
        logits = model(**{**t, **i}).logits
        if logits.dim() == 4: logits = logits.squeeze(1)
        if logits.dim() == 3: logits = logits.squeeze(0)
        prob = torch.sigmoid(logits)
        print(f"  [sanity] max={prob.max():.4f}  "
              f"mean={prob.mean():.4f}  "
              f">0.5: {(prob>0.5).sum().item()} px")

# ── SAVE ─────────────────────────────────────────────────────────

torch.save(model.state_dict(), SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")
print("\nAll done!")
print(f"\nNext — run test_clipseg.py with:")
print(f"  CRACK_IMAGE_DIR = r'{OUTPUT_BASE}\\test\\images'")
print(f"  CRACK_MASK_DIR  = r'{OUTPUT_BASE}\\test\\masks'")
print(f"  MODEL_WEIGHTS   = r'{SAVE_PATH}'")