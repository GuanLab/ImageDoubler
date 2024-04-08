
import os
import cv2
from glob import glob
from bs4 import BeautifulSoup
import pandas as pd

ANNO_PATH = "../../../data/crop_labels1"
IMAGE_PATH = "../../../data/crop_image"
targets = pd.read_csv("../../../data/crop_target/labeler1_targets.csv")

os.system("rm Annotations/*")
os.system("rm JPEGImages/*")

for idx, target in targets.iterrows():
    if target["difficult"]:
        continue
    
    print(target["image_id"], target["cell_num"])
    image_id = target["image_id"]
    image_set = target["image_id"].split('_')[0]
    anno_file = f"{ANNO_PATH}/{image_set}/{image_id}.xml"
    image_file = f"{IMAGE_PATH}/{image_set}/{image_id}.png"
    
    os.system(f"cp {anno_file} Annotations/")
    image = cv2.imread(image_file)
    cv2.imwrite(f"JPEGImages/{image_id}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

