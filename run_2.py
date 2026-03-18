import os
import cv2
import random
import shutil

print("SCRIPT STARTED")

###############################################
# FUNCTION : DRAW BOUNDING BOX FROM YOLO LABEL
###############################################

def draw_bbox(image_path, label_path, save_path):

    if not os.path.exists(image_path):
        print("Path does not exist:", image_path)
        return

    os.makedirs(save_path, exist_ok=True)

    images = os.listdir(image_path)

    print("Found", len(images), "images in", image_path)

    for img_name in images:

        img_file = os.path.join(image_path, img_name)
        name = os.path.splitext(img_name)[0]

        label_file = os.path.join(label_path, name + ".txt")

        img = cv2.imread(img_file)

        if img is None:
            continue

        h, w, _ = img.shape

        if os.path.exists(label_file):

            with open(label_file) as f:

                for line in f:

                    values = list(map(float, line.split()))

                    if len(values) < 5:
                        continue

                    cls = values[0]
                    x = values[1]
                    y = values[2]
                    bw = values[3]
                    bh = values[4]

                    x = int(x * w)
                    y = int(y * h)
                    bw = int(bw * w)
                    bh = int(bh * h)

                    x1 = x - bw // 2
                    y1 = y - bh // 2
                    x2 = x + bw // 2
                    y2 = y + bh // 2

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imwrite(os.path.join(save_path,img_name),img)

    print("Saved bbox images to:", save_path)

###############################################
# DATASET 1 : CRACKS
###############################################

print("\nProcessing CRACKS DATASET")

base1 = r"D:\origin\cracks.v1i.yolov8"

for split in ["train","valid","test"]:

    img_path = os.path.join(base1, split, "images")
    lbl_path = os.path.join(base1, split, "labels")

    save_path = os.path.join(base1, split, "bbox_images")

    print("\nProcessing:", split)

    draw_bbox(img_path, lbl_path, save_path)

###############################################
# DATASET 2 : DRYWALL
###############################################

print("\nProcessing DRYWALL DATASET")

base2 = r"D:\origin\Drywall-Join-Detect.v1i.yolov8"

train_img = os.path.join(base2,"train","images")
train_lbl = os.path.join(base2,"train","labels")

val_img = os.path.join(base2,"valid","images")
val_lbl = os.path.join(base2,"valid","labels")

output = r"D:\origin\Drywall_dataset_fixed"

###############################################
# CREATE OUTPUT FOLDERS
###############################################

for split in ["train","val","test"]:

    os.makedirs(os.path.join(output,split,"images"),exist_ok=True)
    os.makedirs(os.path.join(output,split,"labels"),exist_ok=True)

###############################################
# SPLIT TRAIN INTO TRAIN + TEST
###############################################

images = os.listdir(train_img)

print("Total train images:", len(images))

random.shuffle(images)

test_size = int(0.15 * len(images))

test_images = images[:test_size]
train_images = images[test_size:]

###############################################
# COPY TRAIN
###############################################

print("Creating TRAIN split")

for img in train_images:

    name = os.path.splitext(img)[0]

    shutil.copy(
        os.path.join(train_img,img),
        os.path.join(output,"train","images",img)
    )

    shutil.copy(
        os.path.join(train_lbl,name+".txt"),
        os.path.join(output,"train","labels",name+".txt")
    )

###############################################
# COPY TEST
###############################################

print("Creating TEST split")

for img in test_images:

    name = os.path.splitext(img)[0]

    shutil.copy(
        os.path.join(train_img,img),
        os.path.join(output,"test","images",img)
    )

    shutil.copy(
        os.path.join(train_lbl,name+".txt"),
        os.path.join(output,"test","labels",name+".txt")
    )

###############################################
# COPY VALID -> VAL
###############################################

print("Copying VALID to VAL")

for img in os.listdir(val_img):

    name = os.path.splitext(img)[0]

    shutil.copy(
        os.path.join(val_img,img),
        os.path.join(output,"val","images",img)
    )

    shutil.copy(
        os.path.join(val_lbl,name+".txt"),
        os.path.join(output,"val","labels",name+".txt")
    )

###############################################
# DRAW BBOX FOR NEW DRYWALL DATASET
###############################################

print("\nDrawing bounding boxes for drywall dataset")

for split in ["train","val","test"]:

    img_path = os.path.join(output,split,"images")
    lbl_path = os.path.join(output,split,"labels")

    save_path = os.path.join(output,split,"bbox_images")

    print("Processing drywall:",split)

    draw_bbox(img_path,lbl_path,save_path)

print("\nALL DATASETS READY")