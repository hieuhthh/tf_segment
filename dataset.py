import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

def build_decoder(with_labels=True, target_size=(256, 256)):
    def decode_img_preprocess(img):
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_img(path, is_gray=False):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img = tf.image.rgb_to_grayscale(img) if is_gray else img
        img = decode_img_preprocess(img)
        return img
        
    def decode_with_labels(path_img, path_mask):
        return decode_img(path_img, is_gray=False), decode_img(path_mask, is_gray=True)
    
    return decode_with_labels if with_labels else decode_img

def build_dataset(paths, labels=None, bsize=32,
                  decode_fn=None, augment=None,
                  repeat=False, shuffle=1024,
                  cache=False, cache_dir=""):
    """
    paths: paths to images
    labels: int label
    """              
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((paths))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    # dset = dset.map(augment, num_parallel_calls=AUTO) if augment is not None else dset
    # dset = dset.map(lambda x,y:(augment(x),y), num_parallel_calls=AUTO) if augment is not None else dset
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

def build_dataset_from_X_Y(X_path, Y_int, with_labels, img_size,
                           batch_size, repeat, shuffle, augment):
    decoder = build_decoder(with_labels, img_size)

    # augment_img = build_augment() if augment else None
    augment_img = None

    dataset = build_dataset(X_path, Y_int, bsize=batch_size, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle, augment=augment_img)

    return dataset

if __name__ == '__main__':
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

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

    print(len(train_img_paths))
    print(len(valid_img_paths))
    print(train_img_paths[0])
    print(train_mask_paths[0])
    print(valid_img_paths[0])
    print(valid_mask_paths[0])

    train_n_images = len(train_img_paths)
    train_dataset = build_dataset_from_X_Y(train_img_paths, train_mask_paths, train_with_labels, img_size,
                                           BATCH_SIZE, train_repeat, train_shuffle, train_augment)

    valid_n_images = len(valid_img_paths)
    valid_dataset = build_dataset_from_X_Y(valid_img_paths, valid_mask_paths, valid_with_labels, img_size,
                                           BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

    for x, y in train_dataset:
        break
    print(x)
    print(y)

    import cv2
    import numpy as np
    cv2.imwrite("sample_img.png", np.array(x[0][...,::-1])*255)
    cv2.imwrite("sample_mask.png", np.array(y[0][...,::-1])*255)

