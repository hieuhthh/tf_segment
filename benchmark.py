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

weight_model = 'best_model_segment_256_1.h5'

new_model_save = f'clean_{weight_model}'

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]="3"

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

seedEverything(seed)

n_labels = 1
print('n_labels', n_labels)

strategy = auto_select_accelerator()

with strategy.scope():
    model = create_model(im_size, n_labels, config_map, do_dim, final_dim)
    
    model.summary()

model.load_weights(weight_model)
print('Loaded pretrain from', weight_model)

with strategy.scope():
    new_model = Model(model.input, model.get_layer('output_x1').output)
    new_model.summary()
    new_model.save(new_model_save)

    losses = bce_dice_loss

    metrics = [dice_coeff,
               round_dice_coeff, 
               metric_iou,
               tf.keras.metrics.MeanAbsoluteError()]

    new_model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss=losses,
                      metrics=metrics)

valid_route = 'unzip/polyp/TestDataset'
# for valid_data in os.listdir(valid_route):
for valid_data in ['Kvasir', 'CVC-ClinicDB']:
    valid_path = path_join(valid_route, valid_data)
    valid_img_dir = path_join(valid_path, 'images')
    valid_mask_dir = path_join(valid_path, 'masks')

    valid_img_paths = sorted(glob(valid_img_dir + '/*'))
    valid_mask_paths = sorted(glob(valid_mask_dir + '/*'))

    valid_n_images = len(valid_img_paths)
    valid_dataset = build_dataset_from_X_Y(valid_img_paths, valid_mask_paths, valid_with_labels, img_size,
                                           1, valid_repeat, valid_shuffle, valid_augment, False)

    print('*'*30)
    print(valid_data)
    print('*'*30)

    print("len:", valid_n_images)

    his = new_model.evaluate(valid_dataset)
    print(his)

    with open(f"benchmark_{valid_data}.txt", mode='w') as f:
        for item in his:
            f.write(str(item) + " ")

# Valid train dataset

# valid_route = 'unzip/polyp'

# valid_data = 'TrainDataset'
# valid_path = path_join(valid_route, valid_data)
# valid_img_dir = path_join(valid_path, 'images')
# valid_mask_dir = path_join(valid_path, 'masks')

# valid_img_paths = sorted(glob(valid_img_dir + '/*'))
# valid_mask_paths = sorted(glob(valid_mask_dir + '/*'))

# valid_n_images = len(valid_img_paths)
# valid_dataset = build_dataset_from_X_Y(valid_img_paths, valid_mask_paths, valid_with_labels, img_size,
#                                        VALID_BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment, False)

# print('*'*30)
# print(valid_data)
# print('*'*30)

# print("len:", valid_n_images)

# his = new_model.evaluate(valid_dataset)
# print(his)

# with open(f"benchmark_{valid_data}.txt", mode='w') as f:
#     for item in his:
#         f.write(str(item) + " ")