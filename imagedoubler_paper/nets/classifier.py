import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.initializers import random_normal
from keras.layers import Dense, Flatten, TimeDistributed

from nets.resnet import resnet50_classifier_layers

class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)

        # batch_size, 38, 38, 1024
        feature_map = x[0]
        
        # batch_size, num_roi, 4
        rois        = x[1]
        
        num_rois    = tf.shape(rois)[1]
        batch_size  = tf.shape(rois)[0]
        
        # crop and resize to align the feature shapes in the RoIs 
        box_index   = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index   = tf.tile(box_index, (1, num_rois))
        box_index   = tf.reshape(box_index, [-1])

        rs          = tf.image.crop_and_resize(feature_map, tf.reshape(rois, [-1, 4]), box_index, (self.pool_size, self.pool_size))
            
        # batch_size, num_roi, pool_size, pool_size, 1024
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output

def get_resnet50_classifier(base_layers, input_rois, roi_size, num_classes=21):
    # batch_size, 38, 38, 1024 -> batch_size, num_rois, 14, 14, 1024
    out_roi_pool = RoiPoolingConv(roi_size)([base_layers, input_rois])

    # batch_size, num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = resnet50_classifier_layers(out_roi_pool)

    # batch_size, num_rois, 1, 1, 2048 -> batch_size, num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # batch_size, num_rois, 2048 -> batch_size, num_rois, num_classes
    out_class   = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=random_normal(stddev=0.02)), name='dense_class_{}'.format(num_classes))(out)
    # batch_size, num_rois, 2048 -> batch_size, num_rois, 4 * (num_classes-1)
    out_regr    = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=random_normal(stddev=0.02)), name='dense_regress_{}'.format(num_classes))(out)
    return [out_class, out_regr]
