# import cv2
# import os

# image_path = r"D:\origin\Drywall-Join-Detect.v1i.yolov8\train\images"
# label_path = r"D:\origin\Drywall-Join-Detect.v1i.yolov8\train\labels"

# for img_name in os.listdir(image_path):

#     img = cv2.imread(os.path.join(image_path,img_name))
#     h,w,_ = img.shape

#     label_file = os.path.join(label_path,img_name.replace(".jpg",".txt"))

#     if not os.path.exists(label_file):
#         continue

#     with open(label_file) as f:
#         for line in f:

#             cls,x,y,bw,bh = map(float,line.split())

#             x = int(x*w)
#             y = int(y*h)
#             bw = int(bw*w)
#             bh = int(bh*h)

#             x1 = x - bw//2
#             y1 = y - bh//2
#             x2 = x + bw//2
#             y2 = y + bh//2

#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

#     cv2.imshow("bbox", img)
#     cv2.waitKey(0)




import cv2
import os

image_path = r"D:\origin\Drywall-Join-Detect.v1i.yolov8\train\images"
label_path = r"D:\origin\Drywall-Join-Detect.v1i.yolov8\train\labels"
save_path = r"D:\origin\bbox_visualized"

os.makedirs(save_path, exist_ok=True)

for img_name in os.listdir(image_path):

    img = cv2.imread(os.path.join(image_path, img_name))
    h, w, _ = img.shape

    label_file = os.path.join(label_path, img_name.replace(".jpg", ".txt"))

    if os.path.exists(label_file):

        with open(label_file) as f:
            for line in f:

                cls, x, y, bw, bh = map(float, line.split())

                x = int(x * w)
                y = int(y * h)
                bw = int(bw * w)
                bh = int(bh * h)

                x1 = x - bw // 2
                y1 = y - bh // 2
                x2 = x + bw // 2
                y2 = y + bh // 2

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # save image
    save_file = os.path.join(save_path, img_name)
    cv2.imwrite(save_file, img)

print("All images saved with bounding boxes.")