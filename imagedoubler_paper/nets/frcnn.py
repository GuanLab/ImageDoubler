from keras.layers import Input
from keras.models import Model

from nets.classifier import get_resnet50_classifier
from nets.resnet import ResNet50
from nets.rpn import get_rpn

def get_model(num_classes, backbone, num_anchors = 9):
    inputs      = Input(shape=(None, None, 3))
    roi_input   = Input(shape=(None, 4))
    
    base_layers = ResNet50(inputs)
    rpn         = get_rpn(base_layers, num_anchors)
    classifier  = get_resnet50_classifier(base_layers, roi_input, 14, num_classes)

    model_rpn   = Model(inputs, rpn)
    model_all   = Model([inputs, roi_input], rpn + classifier)
    return model_rpn, model_all

def get_predict_model(num_classes, backbone, num_anchors = 9):
    inputs              = Input(shape=(None, None, 3))
    roi_input           = Input(shape=(None, 4))
    
    feature_map_input = Input(shape=(None, None, 1024))
    
    base_layers = ResNet50(inputs)
    rpn         = get_rpn(base_layers, num_anchors)
    classifier  = get_resnet50_classifier(feature_map_input, roi_input, 14, num_classes)
    
    model_rpn   = Model(inputs, rpn + [base_layers])
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
