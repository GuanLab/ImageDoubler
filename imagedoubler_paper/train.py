import sys
import warnings
import numpy as np
warnings.simplefilter(action="ignore", category=FutureWarning)

import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from nets.frcnn import get_model
from nets.frcnn_training import (ProposalTargetCreator, classifier_cls_loss,
                                 classifier_smooth_l1, rpn_cls_loss,
                                 rpn_smooth_l1)
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch

save_to = sys.argv[1]  # e.g. loocv/Image7
model_id = int(sys.argv[2])


classes_path    = 'model_data/class.txt'
model_path      = 'model_data/voc_weights_resnet.h5'
input_shape     = [600, 600]
backbone        = "resnet50"  # will always use resnet50
anchors_size    = [10, 20, 40]

# freeze the resnet-50 part
Init_Epoch          = 0
Freeze_Epoch        = 10
Freeze_batch_size   = 16
Freeze_lr           = 1e-4

# unfreeze the whole network
UnFreeze_Epoch      = 30
Unfreeze_batch_size = 8
Unfreeze_lr         = 5e-5

Freeze_Train        = True

train_annotation_path   = f'train_val_split/{save_to}/2007_train_{model_id}.txt'
val_annotation_path     = f'train_val_split/{save_to}/2007_val_{model_id}.txt'

# get class and anchors
class_names, num_classes = get_classes(classes_path)
num_classes += 1  # cell and nothing
anchors = get_anchors(input_shape, backbone, anchors_size)

K.clear_session()
model_rpn, model_all = get_model(num_classes, backbone = backbone)
if model_path != '':
    print('Load weights {}.'.format(model_path))
    model_rpn.load_weights(model_path, by_name=True)
    model_all.load_weights(model_path, by_name=True)

dir_for_weight_logs = f"logs/{save_to}"

callback        = TensorBoard(log_dir=dir_for_weight_logs)
callback.set_model(model_all)
loss_history    = LossHistory(dir_for_weight_logs)

bbox_util       = BBoxUtility(num_classes)
roi_helper      = ProposalTargetCreator(num_classes)

# load the samples info
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines   = f.readlines()
num_train   = len(train_lines)
num_val     = len(val_lines)

freeze_layers = 141  # {'vgg' : 17, 'resnet50' : 141}[backbone]
if Freeze_Train:
    for i in range(freeze_layers): 
        if type(model_all.layers[i]) != keras.layers.BatchNormalization:
            model_all.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))


# train the partially freezed network
best_val_loss = 10

if True:
    batch_size  = Freeze_batch_size
    lr          = Freeze_lr
    start_epoch = Init_Epoch
    end_epoch   = Freeze_Epoch

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('Not enough data')

    model_rpn.compile(
        loss = {
            'classification': rpn_cls_loss(),
            'regression'    : rpn_smooth_l1()
        }, optimizer = Adam(lr=lr)
    )
    model_all.compile(
        loss = {
            'classification'                        : rpn_cls_loss(),
            'regression'                            : rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
        }, optimizer = Adam(lr=lr)
    )

    gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
    gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for epoch in range(start_epoch, end_epoch):
        train_loss, val_loss = fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                anchors, bbox_util, roi_helper)
        
        if val_loss < best_val_loss:  # save the weights for best validation loss
            model_all.save_weights(f'{dir_for_weight_logs}/model{model_id}_best_val_loss_weights.h5')
            best_val_loss = val_loss
        if (epoch+1) % 10 == 0:  # save the weights every 10 epochs
            model_all.save_weights(f'{dir_for_weight_logs}/model{model_id}_ep{epoch:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}.h5')
        
        lr = lr*0.96
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)

if Freeze_Train:
    for i in range(freeze_layers): 
        if type(model_all.layers[i]) != keras.layers.BatchNormalization:
            model_all.layers[i].trainable = True

# train the whole network
if True:
    batch_size  = Unfreeze_batch_size
    lr          = Unfreeze_lr
    start_epoch = Freeze_Epoch
    end_epoch   = UnFreeze_Epoch

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('Not enought data')

    model_rpn.compile(
        loss = {
            'classification': rpn_cls_loss(),
            'regression'    : rpn_smooth_l1()
        }, optimizer = Adam(lr=lr)
    )
    model_all.compile(
        loss = {
            'classification'                        : rpn_cls_loss(),
            'regression'                            : rpn_smooth_l1(),
            'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
            'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
        }, optimizer = Adam(lr=lr)
    )

    gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
    gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    for epoch in range(start_epoch, end_epoch):
        train_loss, val_loss = fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                anchors, bbox_util, roi_helper)
        
        if val_loss < best_val_loss:  # save the weights for best validation loss
            model_all.save_weights(f'{dir_for_weight_logs}/model{model_id}_best_val_loss_weights.h5')
            best_val_loss = val_loss
        if (epoch+1) % 10 == 0:  # save the weights every 10 epochs
            model_all.save_weights(f'{dir_for_weight_logs}/model{model_id}_ep{epoch:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}.h5')
        
        lr = lr*0.96
        K.set_value(model_rpn.optimizer.lr, lr)
        K.set_value(model_all.optimizer.lr, lr)
