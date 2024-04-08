
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import json
import os
import imutils
from imutils import paths
from PIL import Image


def crop_images(img_dir):
    img_files = glob(f"../data/blocks/{img_dir}/Image*")
    if not os.path.isdir(f"../data/crop_image/{img_dir}"):
        os.system(f"mkdir -p crop_image/{img_dir}")
        
    template = cv.imread(f"../data/templates/{img_dir}.png", cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    for f in img_files:
        name = os.path.basename(f)
        img = cv.imread(f, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        top_left = max_loc
        img = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        
        cv.imwrite(f"../data/crop_image/{img_dir}/{name}", img)

crop_images("Image1")
crop_images("Image2")
crop_images("Image3")
crop_images("Image5")
crop_images("Image6")
crop_images("Image7")
crop_images("Image8")
crop_images("Image9")
crop_images("Image10")
crop_images("Image11")

# need further hand curation on the failure ones