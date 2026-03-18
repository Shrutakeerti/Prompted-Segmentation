import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

model.load_state_dict(torch.load("clipseg_drywall.pth"))

model.to(device)
model.eval()

image_folder = "datasets/cracks/test/images"

save_folder = "predictions"

os.makedirs(save_folder,exist_ok=True)

prompt = "segment crack"

for img_name in os.listdir(image_folder):

    img_path = os.path.join(image_folder,img_name)

    image = Image.open(img_path).convert("RGB")

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

    mask = outputs.logits.squeeze().cpu().numpy()

    mask = (mask>0).astype(np.uint8)*255

    save_name = os.path.splitext(img_name)[0] + "__segment_crack.png"

    cv2.imwrite(os.path.join(save_folder,save_name),mask)