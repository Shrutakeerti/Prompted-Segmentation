"""
train_cracks.py  —  Fine-tunes CLIPSeg on Cracks dataset (5 epochs)
"""

import os
import random
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

IMAGE_DIR = r"D:\origin\cracks_split\train\images"
MASK_DIR  = r"D:\origin\cracks_split\train\masks"
SAVE_PATH = r"D:\origin\clipseg_cracks.pth"

PROMPTS = [
    "segment crack",
    "segment wall crack",
    "segment surface crack",
]

BATCH  = 4
EPOCHS = 5
LR     = 5e-6
SEED   = 42

# ─────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {device}")
print(f"Images  : {IMAGE_DIR}")
print(f"Save to : {SAVE_PATH}")

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def find_mask(mask_dir, stem):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────

print("\nLoading model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)
print("Model loaded.")

# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class CrackDataset(Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.mask_tf  = T.Compose([T.Resize((352, 352)), T.ToTensor()])

        all_images = [f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        self.images = []
        skipped = 0
        for f in all_images:
            if find_mask(mask_dir, Path(f).stem):
                self.images.append(f)
            else:
                skipped += 1

        if skipped:
            print(f"  Skipped {skipped} images (no mask)")
        print(f"  Dataset: {len(self.images)} images ready")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name  = self.images[idx]
        stem  = Path(name).stem

        image = Image.open(os.path.join(self.img_dir, name)).convert("RGB")

        mask  = Image.open(find_mask(self.mask_dir, stem)).convert("L")
        mask  = self.mask_tf(mask)
        mask  = (mask > 0).float().squeeze(0)    # [352, 352]

        return image, mask, random.choice(PROMPTS)


def collate_fn(batch):
    return (
        [b[0] for b in batch],
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch]
    )

# ─────────────────────────────────────────────────────────────────
# DATALOADER
# ─────────────────────────────────────────────────────────────────

dataset = CrackDataset(IMAGE_DIR, MASK_DIR)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                     num_workers=0, collate_fn=collate_fn)

print(f"\nSamples       : {len(dataset)}")
print(f"Batches/epoch : {len(loader)}")
print(f"Epochs        : {EPOCHS}  |  LR : {LR}  |  Batch : {BATCH}")
print(f"\nStarting... (est. ~{len(loader)*EPOCHS*3//60} min on CPU)\n")

# ─────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for images, masks, prompts in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        masks = masks.to(device)

        # ── KEY FIX: encode text and images separately ──────────────
        # text needs padding=True (prompts have different token lengths)
        # images are processed separately — no padding kwarg needed
        text_inputs  = processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        image_inputs = processor.image_processor(
            images=images,
            return_tensors="pt"
        ).to(device)

        # merge both into one dict for the model
        inputs = {**text_inputs, **image_inputs}

        outputs = model(**inputs)
        preds   = outputs.logits

        if preds.dim() == 4:
            preds = preds.squeeze(1)

        preds = F.interpolate(
            preds.unsqueeze(1),
            size=(352, 352),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)                    # [B, 352, 352]

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

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")
print("Done.")