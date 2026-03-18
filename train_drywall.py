import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

############################################################
# PROMPTS (DRYWALL ONLY)
############################################################

prompts_list = [
    "segment taping area",
    "segment drywall seam"
]

############################################################
# DATASET
############################################################

class SegDataset(Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.images   = os.listdir(img_dir)
        self.resize   = T.Resize((352, 352))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img  = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(
            self.mask_dir,
            os.path.splitext(name)[0] + ".png"
        )).convert("L")

        img  = self.to_tensor(self.resize(img))    # [3, 352, 352]
        mask = self.to_tensor(self.resize(mask))   # [1, 352, 352]
        mask = (mask > 0).float().squeeze(0)       # [352, 352]  ← FIX: remove channel dim here

        prompt = random.choice(prompts_list)

        return img, mask, prompt

############################################################
# LOAD DATA
############################################################

dataset = SegDataset(
    r"D:\origin\Drywall_dataset_fixed\train\images",
    r"D:\origin\Drywall_dataset_fixed\train\masks"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
print("Total samples:", len(dataset))

############################################################
# TRAIN
############################################################

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
loss_fn   = torch.nn.BCEWithLogitsLoss()
to_pil    = T.ToPILImage()
epochs    = 8

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for imgs, masks, prompts in tqdm(loader):

        imgs  = imgs.to(device)   # [B, 3, 352, 352]
        masks = masks.to(device)  # [B, 352, 352]  ← already squeezed in __getitem__

        # CLIPSeg processor expects PIL images
        pil_images = [to_pil(img.cpu()) for img in imgs]

        inputs = processor(
            text=list(prompts),
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)
        preds   = outputs.logits  # [B, H', W'] or [B, 1, H', W']

        # Normalise preds to [B, H', W']
        if preds.dim() == 4:
            preds = preds.squeeze(1)

        # Upsample to 352x352
        preds = F.interpolate(
            preds.unsqueeze(1),   # [B, 1, H', W']
            size=(352, 352),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)              # [B, 352, 352]

        # Both are now [B, 352, 352] — safe
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "clipseg_drywall.pth")
print("\nDrywall model saved as clipseg_drywall.pth")