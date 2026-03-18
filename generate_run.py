import os
import cv2
import numpy as np

print("MASK GENERATION STARTED")

########################################
# FUNCTION TO CREATE MASKS
########################################

def create_masks(image_dir, label_dir, mask_dir):

    os.makedirs(mask_dir, exist_ok=True)

    images = os.listdir(image_dir)

    print("Processing", len(images), "images from", image_dir)

    for img_name in images:

        img_path = os.path.join(image_dir, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        name = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_dir, name + ".txt")

        if os.path.exists(label_file):

            with open(label_file) as f:

                for line in f:

                    values = list(map(float, line.split()))

                    cls = int(values[0])

                    coords = values[1:]

                    polygon = []

                    for i in range(0, len(coords), 2):

                        x = int(coords[i] * w)
                        y = int(coords[i+1] * h)

                        polygon.append([x, y])

                    polygon = np.array(polygon, dtype=np.int32)

                    cv2.fillPoly(mask, [polygon], 255)

        mask_path = os.path.join(mask_dir, name + ".png")

        cv2.imwrite(mask_path, mask)

    print("Masks saved to:", mask_dir)

########################################
# DATASET 1 : CRACKS
########################################

base1 = r"D:\origin\cracks.v1i.yolov8"

for split in ["train","valid","test"]:

    img_dir = os.path.join(base1, split, "images")
    lbl_dir = os.path.join(base1, split, "labels")
    mask_dir = os.path.join(base1, split, "masks")

    print("\nGenerating masks for CRACKS:", split)

    create_masks(img_dir, lbl_dir, mask_dir)

########################################
# DATASET 2 : DRYWALL
########################################

base2 = r"D:\origin\Drywall_dataset_fixed"

for split in ["train","val","test"]:

    img_dir = os.path.join(base2, split, "images")
    lbl_dir = os.path.join(base2, split, "labels")
    mask_dir = os.path.join(base2, split, "masks")

    print("\nGenerating masks for DRYWALL:", split)

    create_masks(img_dir, lbl_dir, mask_dir)

print("\nALL MASKS GENERATED")