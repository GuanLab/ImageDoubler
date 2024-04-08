import math
from random import shuffle

import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

from utils.utils import cvtColor


class FRCNNDatasets():
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, n_sample = 256, ignore_threshold = 0.3, overlap_threshold = 0.7):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.num_anchors        = len(anchors)
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train
        self.n_sample           = n_sample
        self.ignore_threshold   = ignore_threshold
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def generate(self):
        i = 0
        while True:
            image_data      = []
            classifications = []
            regressions     = []
            targets         = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(self.annotation_lines)
                
                # only apply augmentation in train
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
                if len(box)!=0:
                    boxes               = np.array(box[:, :4] , dtype=np.float32)
                    boxes[:, [0, 2]]    = boxes[:,[0, 2]] / self.input_shape[1]
                    boxes[:, [1, 3]]    = boxes[:,[1, 3]] / self.input_shape[0]
                    box                 = np.concatenate([boxes, box[:, -1:]], axis=-1)

                assignment  = self.assign_boxes(box)
                classification  = assignment[:, 4]
                regression      = assignment[:, :]

                # filter samples
                pos_index   = np.where(classification > 0)[0]
                if len(pos_index) > self.n_sample / 2:
                    disable_index = np.random.choice(pos_index, size=(len(pos_index) - self.n_sample // 2), replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1
                        
                # balance samples
                n_neg       = self.n_sample - np.sum(classification > 0)
                neg_index   = np.where(classification == 0)[0]
                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
                    classification[disable_index] = -1
                    regression[disable_index, -1] = -1
                    
                i = (i+1) % self.length
                image_data.append(preprocess_input(image))
                classifications.append(np.expand_dims(classification, -1))
                regressions.append(regression)
                targets.append(box)

            yield np.array(image_data), [np.array(classifications,dtype=np.float32), np.array(regressions,dtype=np.float32)], targets

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    # data augmentations
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        image   = Image.open(line[0])
        image   = cvtColor(image)
        
        iw, ih  = image.size
        h, w    = input_shape
        
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            # scale the image to given size, first keep its original ratio, and then pad with constant value
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            # adjust the bounding boxes to the scaled images
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        # augmentation: resize and jitter
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # padding
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # augmentation: random flip
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.float32)
        
        # augmentation: color hue
        # hue = self.rand(-hue, hue)
        # sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        # val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        # x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        # x[..., 0] += hue*360
        # x[..., 0][x[..., 0]>1] -= 1
        # x[..., 0][x[..., 0]<0] += 1
        # x[..., 1] *= sat
        # x[..., 2] *= val
        # x[x[:,:, 0]>360, 0] = 360
        # x[:, :, 1:][x[:, :, 1:]>1] = 1
        # x[x<0] = 0
        # image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # adjust the bounding boxes
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box

    def iou(self, box):
        inter_upleft    = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright  = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh    = inter_botright - inter_upleft
        inter_wh    = np.maximum(inter_wh, 0)
        inter       = inter_wh[:, 0] * inter_wh[:, 1]
        
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_ignore_box(self, box, return_iou=True, variances = [0.25, 0.25, 0.25, 0.25]):
        iou         = self.iou(box)
        ignored_box = np.zeros((self.num_anchors, 1))
        
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        
        assign_mask = iou > self.overlap_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        assigned_anchors = self.anchors[assign_mask]
        box_center  = 0.5 * (box[:2] + box[2:])
        box_wh      = box[2:] - box[:2]
        assigned_anchors_center = 0.5 * (assigned_anchors[:, :2] + assigned_anchors[:, 2:4])
        assigned_anchors_wh     = assigned_anchors[:, 2:4] - assigned_anchors[:, :2]

        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes):
        assignment          = np.zeros((self.num_anchors, 4 + 1))
        assignment[:, 4]    = 0.0
        if len(boxes) == 0:
            return assignment

        apply_along_axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        ingored_boxes   = ingored_boxes.reshape(-1, self.num_anchors, 1)
        ignore_iou      = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1

        encoded_boxes   = encoded_boxes.reshape(-1, self.num_anchors, 5)
        
        best_iou        = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx    = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask   = best_iou > 0
        best_iou_idx    = best_iou_idx[best_iou_mask]
        
        assign_num      = len(best_iou_idx)

        encoded_boxes   = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 1
        return assignment
