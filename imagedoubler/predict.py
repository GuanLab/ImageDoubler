
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

import argparse


parser = argparse.ArgumentParser(description='Arguments for making inference on the images with the trained models')
parser.add_argument('--model-path', type=str, required=True, help='The path to the weight file for loading')
parser.add_argument('--model-id', type=str, default='0', help='The model id of the loaded weights. Default is "0"')
parser.add_argument('--conf', type=float, default=0.5, help='The confidence threshold for the detection. Default is 0.5')
parser.add_argument('--image-dir', type=str, required=True, help='The directory of the images to be inferred')
parser.add_argument('--out-dir', type=str, default="img_out/", help='The directory to save the inference results. Default is img_out/')

args = parser.parse_args()
    
# model_path      = f'logs/{save_to}/model{model_id}_best_val_loss_weights.h5'
model_path      = args.model_path
classes_path    = 'model_data/class.txt'
frcnn = FRCNN(confidence = args.conf, nms_iou = 0.5, model_path = model_path, classes_path = classes_path)

dir_origin_path = args.image_dir
dir_save_path_img = f"{args.out_dir}/detection-results-img-model{args.model_id}/"
dir_save_path_txt = args.out_dir

img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    image_path  = os.path.join(dir_origin_path, img_name)
    image       = Image.open(image_path)
    r_image     = frcnn.detect_image(image)
    if not os.path.exists(dir_save_path_img):
        os.makedirs(dir_save_path_img)
    r_image.save(os.path.join(dir_save_path_img, img_name), quality=95, subsampling=0)
        
class_names, _ = get_classes(classes_path)
for img_name in tqdm(img_names):
    image_id = os.path.splitext(img_name)[0]
    image_path  = os.path.join(dir_origin_path, img_name)
    image       = Image.open(image_path)
    if not os.path.exists(os.path.join(dir_save_path_txt, f'detection-results-model{args.model_id}')):
        os.makedirs(os.path.join(dir_save_path_txt, f'detection-results-model{args.model_id}'))
    frcnn.get_map_txt(image_id, image, class_names, dir_save_path_txt, args.model_id)

    