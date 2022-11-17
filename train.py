import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from dataset import *
from utils import *
from model import *
from losses import *
from callbacks import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

epochs = cycle_epoch * n_cycle
print('epochs:', epochs)

seedEverything(seed)
print('BATCH_SIZE:', BATCH_SIZE)

train_img_paths = glob('unzip/polyp/TrainDataset/images/*')
train_mask_paths = glob('unzip/polyp/TrainDataset/masks/*')

valid_img_paths = []
valid_mask_paths = []

valid_route = 'unzip/polyp/TestDataset'
for valid_data in os.listdir(valid_route):
    valid_path = path_join(valid_route, valid_data)
    valid_img_dir = path_join(valid_path, 'images')
    valid_mask_dir = path_join(valid_path, 'masks')

    valid_img_paths += glob(valid_img_dir + '/*')
    valid_mask_paths += glob(valid_mask_dir + '/*')

train_n_images = len(train_img_paths)
train_dataset = build_dataset_from_X_Y(train_img_paths, train_mask_paths, train_with_labels, img_size,
                                        BATCH_SIZE, train_repeat, train_shuffle, train_augment)

valid_n_images = len(valid_img_paths)
valid_dataset = build_dataset_from_X_Y(valid_img_paths, valid_mask_paths, valid_with_labels, img_size,
                                        BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

n_labels = 1

print('n_labels', n_labels)
print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(im_size, n_labels, use_dim)
    
    model.summary()

    losses = bce_dice_loss

    metrics = [dice_coeff, IoU, zero_IoU, tf.keras.metrics.MeanAbsoluteError()]

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=losses,
                  metrics=metrics)

if pretrained is not None:
    try:
        model.load_weights(pretrained)
        print('Loaded pretrain from', pretrained)
    except:
        print('Failed to load pretrain from', pretrained)

save_path = f'best_model_segment_{im_size}_{n_labels}.h5'

callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch, save_weights_only)

his = model.fit(train_dataset, 
                validation_data=valid_dataset,
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

metric = 'loss'
visual_save_metric(his, metric)

metric = 'dice_coeff'
visual_save_metric(his, metric)

metric = 'IoU'
visual_save_metric(his, metric)

metric = 'mean_absolute_error'
visual_save_metric(his, metric)

# # EVALUATE

# valid_eval = model.evaluate(valid_dataset)

# print("valid_eval", valid_eval)

# with open("valid_eval.txt", mode='w') as f:
#     for item in valid_eval:
#         f.write(str(item) + " ")




