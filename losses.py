import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, Optimizer
import tensorflow_addons as tfa
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

# loss

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

# metric

def dice_coeff(y_true, y_pred):
    # add epsilon to avoid a divide by 0 error in case a slice has no pixels set
    # we only care about relative value, not absolute so this alteration doesn't matter
    _epsilon = 10 ** -7
    intersections = tf.reduce_sum(y_true * y_pred)
    unions = tf.reduce_sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    return dice_scores

def round_dice_coeff(y_true, y_pred):
    # add epsilon to avoid a divide by 0 error in case a slice has no pixels set
    # we only care about relative value, not absolute so this alteration doesn't matter
    y_true = tf.math.round(y_true)
    y_pred = tf.math.round(y_pred)
    _epsilon = 10 ** -7
    intersections = tf.reduce_sum(y_true * y_pred)
    unions = tf.reduce_sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    return dice_scores

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = tf.reshape(y_true,[-1])
    y_pred_pos = tf.reshape(y_pred,[-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def metric_iou(y_true, y_pred, threshold=0.5, epsilon=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.where(y_pred > threshold, x=1.0, y=0.0)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return intersection / (union + epsilon)

# tf.keras.metrics.MeanAbsoluteError

if __name__ == '__main__':
    import os
    from utils import *
    from tf_segment.model import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""
    
    settings = get_settings()
    globals().update(settings)

    n_labels = 1

    model = create_model(im_size, n_labels, config_map, do_dim, final_dim)

    losses = bce_dice_loss

    metrics = [dice_coeff,
               round_dice_coeff, 
               metric_iou,
               tf.keras.metrics.MeanAbsoluteError()]


    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=losses,
                  metrics=metrics)

    model.summary()