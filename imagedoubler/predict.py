
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from frcnn import FRCNN
from utils.utils import get_classes

save_to = sys.argv[1]
model_id = int(sys.argv[2])  # 1 2 3 4 5

    
model_path      = f'logs/{save_to}/model{model_id}_best_val_loss_weights.h5'
classes_path    = 'model_data/class.txt'
frcnn = FRCNN(confidence = 0.7, nms_iou = 0.5, model_path = model_path, classes_path = classes_path)

dir_origin_path = f"img/{save_to}/"
dir_save_path_img = f"img_out/{save_to}/detection-results-img-model{model_id}/"
dir_save_path_txt = f"img_out/{save_to}/"

img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    image_path  = os.path.join(dir_origin_path, img_name)
    image       = Image.open(image_path)
    r_image     = frcnn.detect_image(image)
    if not os.path.exists(dir_save_path_img):
        os.makedirs(dir_save_path_img)
    r_image.save(os.path.join(dir_save_path_img, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
        
class_names, _ = get_classes(classes_path)
for img_name in tqdm(img_names):
    image_id = os.path.splitext(img_name)[0]
    image_path  = os.path.join(dir_origin_path, img_name)
    image       = Image.open(image_path)
    if not os.path.exists(os.path.join(dir_save_path_txt, f'detection-results-model{model_id}')):
        os.makedirs(os.path.join(dir_save_path_txt, f'detection-results-model{model_id}'))
    frcnn.get_map_txt(image_id, image, class_names, dir_save_path_txt, model_id)

    