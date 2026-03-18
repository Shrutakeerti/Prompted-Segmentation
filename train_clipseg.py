import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F

############################################################
# DEVICE
############################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################################
# LOAD MODEL
############################################################

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)

############################################################
# PROMPTS
############################################################

crack_prompts = [
    "segment crack",
    "segment wall crack"
]

drywall_prompts = [
    "segment taping area",
    "segment joint tape",
    "segment drywall seam"
]

############################################################
# DATASET CLASS
############################################################

class SegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, prompts):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.prompts   = prompts
        self.images    = os.listdir(image_dir)

        self.resize    = T.Resize((352, 352))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(img_name)[0] + ".png"
        )

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.to_tensor(self.resize(image))   # [3, 352, 352]
        mask  = self.to_tensor(self.resize(mask))    # [1, 352, 352]
        mask  = (mask > 0).float()                   # binarise
        mask  = mask.squeeze(0)                      # [352, 352]

        prompt = random.choice(self.prompts)
        return image, mask, prompt

############################################################
# LOAD DATASETS
############################################################

print("Loading datasets...")

crack_dataset = SegDataset(
    r"D:\origin\cracks.v1i.yolov8\train\images",
    r"D:\origin\cracks.v1i.yolov8\train\masks",
    crack_prompts
)

drywall_dataset = SegDataset(
    r"D:\origin\Drywall_dataset_fixed\train\images",
    r"D:\origin\Drywall_dataset_fixed\train\masks",
    drywall_prompts
)

dataset = torch.utils.data.ConcatDataset([crack_dataset, drywall_dataset])
loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print("Total training samples:", len(dataset))

############################################################
# TRAINING SETUP
############################################################

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn   = torch.nn.BCEWithLogitsLoss()
epochs    = 5

############################################################
# TRAIN LOOP
############################################################

to_pil = T.ToPILImage()

for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    for images, masks, prompts in tqdm(loader):

        images = images.to(device)
        masks  = masks.to(device)

        # ── Guarantee masks shape is [B, 352, 352] no matter what ─────────
        if masks.dim() == 4:
            masks = masks.squeeze(1)   # [B, 1, H, W] → [B, H, W]

        # ── CLIPSeg processor needs PIL images, not tensors ────────────────
        pil_images = [to_pil(img.cpu()) for img in images]

        inputs = processor(
            text=list(prompts),
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)
        preds   = outputs.logits       # [B, H', W'] or [B, 1, H', W']

        # ── Guarantee preds shape is [B, H', W'] ──────────────────────────
        if preds.dim() == 4:
            preds = preds.squeeze(1)

        # ── Upsample preds to 352x352 ─────────────────────────────────────
        preds = F.interpolate(
            preds.unsqueeze(1),        # [B, 1, H', W']
            size=(352, 352),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)                   # [B, 352, 352]

        # ── Both are [B, 352, 352] now — safe to compute loss ─────────────
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1} — Avg Loss: {avg_loss:.4f}")

############################################################
# SAVE MODEL
############################################################

torch.save(model.state_dict(), "clipseg_drywall_model.pth")
print("\nTraining Finished")
print("Model saved as clipseg_drywall_model.pth")