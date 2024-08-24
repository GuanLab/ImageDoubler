import os
import sys
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import xml.etree.ElementTree as ET

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from frcnn import FRCNN

save_to = sys.argv[1]  # e.g. loocv/Image7
model_id = int(sys.argv[2])  # 1 2 3 4 5


model_path      = f'logs/{save_to}/model{model_id}_best_val_loss_weights.h5'
classes_path    = 'model_data/class.txt'

# mAP{MINOVERLAP}
MINOVERLAP      = 0.5

# visualize the VOC_map calculation
map_vis         = False

VOCdevkit_path  = 'VOCdevkit'
map_out_path    = f'map_out/{save_to}'

image_ids = open(os.path.join(VOCdevkit_path, f"VOC2007/ImageSets/{save_to}/test_{model_id}.txt")).read().strip().split()

if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)
if not os.path.exists(os.path.join(map_out_path, f'detection-results-model{model_id}')):
    os.makedirs(os.path.join(map_out_path, f'detection-results-model{model_id}'))  

class_names, _ = get_classes(classes_path)


print("Load model.")
frcnn = FRCNN(confidence = 0.01, nms_iou = 0.5, model_path = model_path, classes_path = classes_path)
print("Load model done.")

print("Get predict result.")
for image_id in tqdm(image_ids):
    image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
    image       = Image.open(image_path)
    if map_vis:
        image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
    frcnn.get_map_txt(image_id, image, class_names, map_out_path, model_id)
print("Get predict result done.")
    
