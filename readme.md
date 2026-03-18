# Prompted Segmentation for Drywall QA

## Goal

Fine-tune a text-conditioned segmentation model (CLIPSeg) to produce binary masks given an image and a natural-language prompt:

| Prompt                  | Task                                  |
| ----------------------- | ------------------------------------- |
| `"segment crack"`       | Detect cracks in walls                |
| `"segment taping area"` | Detect drywall seams / taping regions |

---

## Repository Structure

```
origin/
├── train_clipseg.py          # Joint training (both datasets)
├── train_drywall.py          # Drywall-only fine-tuning
├── test_clipseg.py           # Evaluation — crack + drywall
├── test_drywall.py           # Evaluation — drywall only
├── generate_masks.py         # Generate GT masks from YOLO labels
├── clipseg_drywall_model.pth # Joint fine-tuned weights (603.2 MB)
├── clipseg_drywall.pth       # Drywall-only fine-tuned weights (603.2 MB)
│
├── cracks.v1i.yolov8/
│   ├── train/
│   │   ├── images/       (5162)
│   │   ├── labels/       (5162)
│   │   ├── bbox_images/  (5162)
│   │   └── masks/        (5162)
│   └── test/
│       ├── images/       (4)
│       ├── labels/       (4)
│       ├── bbox_images/  (4)
│       └── masks/        (4)
│
└── Drywall_dataset_fixed/
    ├── train/
    │   ├── images/       (697)
    │   ├── labels/       (697)
    │   ├── bbox_images/  (697)
    │   └── masks/        (697)
    ├── val/
    │   ├── images/       (202)
    │   ├── labels/       (202)
    │   ├── bbox_images/  (202)
    │   └── masks/        (202)
    └── test/
        ├── images/       (123)
        ├── labels/       (123)
        ├── bbox_images/  (123)
        └── masks/        (123)
```

---

## Datasets

| Dataset             | Source                                                                           | Prompt(s)                                     | Train | Val | Test |
| ------------------- | -------------------------------------------------------------------------------- | --------------------------------------------- | ----- | --- | ---- |
| Drywall Join Detect | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | `segment taping area`, `segment drywall seam` | 697   | 202 | 123  |
| Cracks              | [Roboflow](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)                 | `segment crack`, `segment wall crack`         | 5162  | —   | 4    |

> **Note:** The Cracks dataset originally had no val split. The test split contains only 4 images. All 5162 training images were used for fine-tuning.

---

## Dataset Preparation Challenges

Setting up the datasets was **the most time-consuming part of this project**. The following issues were encountered and resolved:

### 1. Missing Val Split (Cracks Dataset)

The Cracks dataset from Roboflow was downloaded with only `train` and `test` folders — no `val` split. The test set contained only 4 images, making proper validation impossible without restructuring.

### 2. No Mask Folder Provided

Neither dataset came with segmentation masks — only bounding-box label files in YOLO format (`.txt`). Masks had to be generated from scratch using `generate_masks.py`, which reads each label file and draws filled bounding boxes as binary masks `{0, 255}`.

### 3. Incorrect / Misaligned Masks (First Attempt)

The first version of the training masks was generated incorrectly — the mask filenames did not match the image filenames due to Roboflow's naming convention (e.g., `img_jpg.rf.xxxxxx.jpg`). The model trained on these misaligned masks produced near-zero probability outputs (`prob max ~0.007`) and IoU of 0.000 across all images.

### 4. Wrong Image Input to Processor

During early training, images were converted to tensors first, then passed through `ToPILImage()` before being fed to CLIPSegProcessor. This introduced subtle pixel value range errors (float 0-1 instead of uint8 0-255), causing the model to receive corrupted inputs and learn nothing. The fix was to pass raw PIL images directly to the processor.

### 5. Shape Mismatch During Training

The mask tensor from `ToTensor()` had shape `[1, 352, 352]` but predictions were `[B, 352, 352]` after batching. This caused repeated `BCEWithLogitsLoss` shape mismatch errors across multiple training attempts. Fixed by calling `.squeeze(0)` on the mask inside `__getitem__`.

### 6. bbox_images Folder Creation

The original dataset had only `images/` and `labels/`. The `bbox_images/` folder was created as an additional step — containing images with bounding boxes drawn on them for visualization and mask verification.

---

## Model

**Base model:** [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)

CLIPSeg uses CLIP vision + text encoders with a lightweight FiLM decoder to produce segmentation masks conditioned on free-form text prompts.

**Fine-tuning config:**

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Input size    | 352 x 352                 |
| Loss          | BCEWithLogitsLoss         |
| Optimizer     | Adam                      |
| Learning rate | 5e-6                      |
| Batch size    | 4                         |
| Epochs        | 8-10 (drywall), 5 (joint) |
| Seed          | 42                        |

---

## Reproducing Results

### 1. Install dependencies

```bash
conda create -n drywall_seg python=3.10
conda activate drywall_seg
pip install torch torchvision transformers pillow tqdm
```

### 2. Generate masks from YOLO labels

```bash
python generate_masks.py
```

### 3. Train

```bash
python train_drywall.py       # drywall-only
python train_clipseg.py       # joint (both datasets)
```

### 4. Evaluate

```bash
python test_drywall.py        # results -> D:\origin\results_drywall\
python test_clipseg.py        # results -> D:\origin\results\
```

---

## Prediction Masks

- Format: PNG, single-channel, same spatial size as source image
- Values: `{0, 255}`
- Filename: `{image_stem}__{prompt_with_underscores}.png`
  - e.g. `img_001__segment_crack.png`
  - e.g. `img_045__segment_taping_area.png`

---

## Output Files

| File          | Description                              |
| ------------- | ---------------------------------------- |
| `report.html` | Full HTML report — open in browser       |
| `summary.txt` | Plain-text metrics table                 |
| `metrics.csv` | Per-image IoU & Dice — open in Excel     |
| `masks/`      | All prediction mask PNGs                 |
| `visuals/`    | Side-by-side: Original / GT / Prediction |

---

## Seeds

All scripts use seed **42**:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
```

---

## Hardware & Runtime

| Item                                         | Value                 |
| -------------------------------------------- | --------------------- |
| Device                                       | CPU (Intel)           |
| Model size                                   | 603.2 MB              |
| Avg inference time                           | ~700 ms / image (CPU) |
| Train time — Drywall (10 epochs, 697 images) | ~3-4 hrs (CPU)        |
| Train time — Joint (5 epochs, 5859 images)   | ~6-8 hrs (CPU)        |

> GPU strongly recommended. CPU training is functional but slow.
