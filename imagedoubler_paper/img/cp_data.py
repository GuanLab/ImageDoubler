
import os
import cv2
from glob import glob

IMAGE_PATH = "../../data/crop_image"

# for expression
for image_file in glob(f"{IMAGE_PATH}/Image5/*.png"):
    image_id = os.path.splitext(os.path.basename(image_file))[0]
    image = cv2.imread(image_file)
    cv2.imwrite(f"for_expression/{image_id}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

for image_file in glob(f"{IMAGE_PATH}/Image11/*.png"):
    image_id = os.path.splitext(os.path.basename(image_file))[0]
    image = cv2.imread(image_file)
    cv2.imwrite(f"for_expression/{image_id}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

for image_set in ["Image" + str(i) for i in [1,2,3,5,6,7,8,9,10,11]]:
    for image_file in glob(f"{IMAGE_PATH}/{image_set}/*.png"):
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        image = cv2.imread(image_file)
        cv2.imwrite(f"loocv/{image_set}/{image_id}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])