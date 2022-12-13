import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import math
import os

from utils import *

def clip_image(image):
    image -= tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    if image_max != 0:
        image = image / image_max
    return image

def cutout_seg(image, mask, DIM, PROBABILITY = 1.0, CT = 1, SZ = 0.4):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE

    P = tf.cast(tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    if (P==0)|(CT==0)|(SZ==0): return image, mask
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast(SZ*DIM,tf.int32)
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        onem = mask[ya:yb,0:xa,:]
        
        two = tf.zeros([yb-ya,xb-xa,3]) 
        twom = tf.zeros([yb-ya,xb-xa,1]) 
    
        three = image[ya:yb,xb:DIM,:]
        threem = mask[ya:yb,xb:DIM,:]
        
        middle = tf.concat([one,two,three],axis=1)
        middlem = tf.concat([onem,twom,threem],axis=1)
        
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)
        mask = tf.concat([mask[0:ya,:,:],middlem,mask[yb:DIM,:,:]],axis=0)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 
    image = tf.reshape(image,[DIM,DIM,3])
    mask = tf.reshape(mask,[DIM,DIM,1])
    return image, mask

def build_augment():
    setting_cfg = get_settings('setting.yaml')
    aug_cfg = get_settings('augment.yaml')

    im_size = setting_cfg['im_size']

    def augment_image_mask(image, mask):
        # Color
        P = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['color_prob'], tf.int32)
        if P == 1:
            image = tf.image.random_hue(image, aug_cfg['hue'])
            image = tf.image.random_saturation(image, aug_cfg['sature_lower'], aug_cfg['sature_upper'])
            image = tf.image.random_contrast(image, aug_cfg['contrast_lower'], aug_cfg['contrast_upper'])
            image = tf.image.random_brightness(image, aug_cfg['bri'])

            if aug_cfg['jpg_quality_lower'] is not None and aug_cfg['jpg_quality_upper'] is not None:
                image = tf.image.random_jpeg_quality(image, aug_cfg['jpg_quality_lower'], aug_cfg['jpg_quality_upper'])
        
        # Random Crop
        P = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['crop_prob'], tf.int32)
        if P == 1:
            offset_x = tf.random.uniform([], 0, tf.cast(im_size * aug_cfg['crop_rate'], tf.int32), dtype=tf.int32)

            max_img_size_crop = im_size - offset_x
            img_size_crop = tf.random.uniform([], tf.cast(tf.cast(max_img_size_crop, tf.float32) * (1.0 - aug_cfg['crop_rate']), tf.int32), max_img_size_crop, dtype=tf.int32)

            if offset_x > 0:
                offset_y = tf.random.uniform([], 0, offset_x, dtype=tf.int32)
            else:
                offset_y = tf.constant(0, dtype=tf.int32)
                
            image = tf.slice(image, [offset_x, offset_y, 0], [img_size_crop, img_size_crop, 3])
            mask = tf.slice(mask, [offset_x, offset_y, 0], [img_size_crop, img_size_crop, 1])
            
            image = tf.image.resize(image, [im_size, im_size], method=tf.image.ResizeMethod.BICUBIC)            
            mask = tf.image.resize(mask, [im_size, im_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # Rotate
        P = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['rot_prob'], tf.int32)
        if P == 1:
            angle = tf.random.uniform([], -aug_cfg['rot_angle'] * math.pi / 180, aug_cfg['rot_angle'] * math.pi / 180, dtype=tf.float32)
            image = tfa.image.rotate(image, angle, interpolation='bilinear', fill_mode='reflect')
            mask = tfa.image.rotate(mask, angle, interpolation='nearest', fill_mode='reflect')
        
        # Cutout
        image, mask = cutout_seg(image, mask, im_size, aug_cfg['cutout_prob'], aug_cfg['cutout_n'], aug_cfg['cutout_size'])

        # Rescale
        image = clip_image(image)
        mask = clip_image(mask)
        
        # Explicit Reshape for TPU
        # image = tf.reshape(image, [im_size, im_size, 3])
        # mask = tf.reshape(mask, [im_size, im_size, 1])

        return image, mask

    return augment_image_mask

def batch_mixup_seg(images, masks, batch_size, PROBABILITY=1.0):
    # Do `batch_mixup` with a probability = `PROBABILITY`
    # This is a tensor containing 0 or 1 -- 0: no mixup.
    # shape = [batch_size]
    do_mixup = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # ratio of importance of the 2 images to be mixed up
    # shape = [batch_size]
    a = tf.random.uniform([batch_size], 0, 1) * tf.cast(do_mixup, tf.float32)  # this is beta dist with alpha=1.0
                
    # The second part corresponds to the images to be added to the original images `images`.
    new_images =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new_image_indices)

    # Make masks
    new_masks =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * masks + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(masks, new_image_indices)

    return new_images, new_masks

def batch_cutmix_seg(images, labels, DIM, batch_size, PROBABILITY=1.0):
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    
    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    new_y = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    
    # Random width for new images, shape = [batch_size]
    b = tf.random.uniform([batch_size], 0, 1) # this is beta dist with alpha=1.0
    new_width = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32) * do_cutmix
    
    # shape = [batch_size]
    new_y0 = tf.math.maximum(0, new_y - new_width // 2)
    new_y1 = tf.math.minimum(DIM, new_y + new_width // 2)
    new_x0 = tf.math.maximum(0, new_x - new_width // 2)
    new_x1 = tf.math.minimum(DIM, new_x + new_width // 2)
    
    # shape = [batch_size, DIM]
    target = tf.broadcast_to(tf.range(DIM), shape=(batch_size, DIM))
    
    # shape = [batch_size, DIM]
    mask_y = tf.math.logical_and(new_y0[:, tf.newaxis] <= target, target <= new_y1[:, tf.newaxis])
    
    # shape = [batch_size, DIM]
    mask_x = tf.math.logical_and(new_x0[:, tf.newaxis] <= target, target <= new_x1[:, tf.newaxis])    
    
    # shape = [batch_size, DIM, DIM]
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

    # All components are of shape [batch_size, DIM, DIM, 3]
    new_images =  images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3]) + \
                    tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3])

    a = tf.cast(new_width ** 2 / DIM ** 2, tf.float32)    
        
    # Make labels     
        
    new_labels =  labels * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 1]) + \
                    tf.gather(labels, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 1])
        
    return new_images, new_labels

def batch_mosaic_seg(x, y, h, w, batch_size, range_mosaic):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mosaic applied
    # 0 <= range <= 0.5
    
    x_list, y_list = [], []
    range_lower = 0.5-range_mosaic
    range_upper = 0.5+range_mosaic
  
    for i in range(batch_size):
        xr = tf.random.uniform([], range_lower, range_upper)
        yr = tf.random.uniform([], range_lower, range_upper)
        
        x_center = tf.cast(tf.math.multiply(xr,tf.cast(w, tf.float32)), tf.int32)
        y_center = tf.cast(tf.math.multiply(yr,tf.cast(h, tf.float32)), tf.int32)

        batch_idx_mosaic0 = i
        batch_idx_mosaic1 = tf.random.uniform([], 0, batch_size, tf.int32)
        batch_idx_mosaic2 = tf.random.uniform([], 0, batch_size, tf.int32)
        batch_idx_mosaic3 = tf.random.uniform([], 0, batch_size, tf.int32)

        x1_shape = x.shape[1:]
        x1 = tf.concat([tf.concat([x[batch_idx_mosaic0][:y_center, :x_center, :], 
                                  x[batch_idx_mosaic1][:y_center, x_center:, :]], axis=1), 
                        tf.concat([x[batch_idx_mosaic2][y_center:, :x_center, :], 
                                   x[batch_idx_mosaic3][y_center:, x_center:, :]], axis=1)], axis=0)
        x1 = tf.reshape(x1, x1_shape)
        
        x2_shape = y.shape[1:]
        x2 = tf.concat([tf.concat([y[batch_idx_mosaic0][:y_center, :x_center, :], 
                                  y[batch_idx_mosaic1][:y_center, x_center:, :]], axis=1), 
                        tf.concat([y[batch_idx_mosaic2][y_center:, :x_center, :], 
                                   y[batch_idx_mosaic3][y_center:, x_center:, :]], axis=1)], axis=0)
        x2 = tf.reshape(x2, x2_shape)
        
        x_list.append(x1)
        y_list.append(x2)
        
    x, y = tf.stack(x_list, axis=0), tf.stack(y_list, axis=0)
    return x, y

def batch_mosaic_full_seg(x, y, batch_size, range_mosaic):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mosaic full applied
    
    x_list, y_list = [], []
    range_lower = 0.5-range_mosaic
    range_upper = 0.5+range_mosaic

    for i in range(batch_size):
        x_center = tf.cast(tf.random.uniform([], range_lower, range_upper) * x.shape[2], tf.int32)
        y_center = tf.cast(tf.random.uniform([], range_lower, range_upper) * x.shape[1], tf.int32)

        batch_idx_mosaic0 = i
        batch_idx_mosaic1 = tf.random.uniform([], 0, batch_size, tf.int32)
        batch_idx_mosaic2 = tf.random.uniform([], 0, batch_size, tf.int32)
        batch_idx_mosaic3 = tf.random.uniform([], 0, batch_size, tf.int32)

        img0 = tf.image.resize(x[batch_idx_mosaic0], (y_center, x_center))
        img1 = tf.image.resize(x[batch_idx_mosaic1], (y_center, x.shape[2] - x_center))
        img2 = tf.image.resize(x[batch_idx_mosaic2], (x.shape[1] - y_center, x_center))
        img3 = tf.image.resize(x[batch_idx_mosaic3], (x.shape[1] - y_center, x.shape[2] - x_center))
        
        simg0 = tf.image.resize(y[batch_idx_mosaic0], (y_center, x_center))
        simg1 = tf.image.resize(y[batch_idx_mosaic1], (y_center, y.shape[2] - x_center))
        simg2 = tf.image.resize(y[batch_idx_mosaic2], (y.shape[1] - y_center, x_center))
        simg3 = tf.image.resize(y[batch_idx_mosaic3], (y.shape[1] - y_center, y.shape[2] - x_center))
        
        x1_shape = x.shape[1:]
        x1 = tf.concat([tf.concat([img0, img1], axis=1), 
                        tf.concat([img2, img3], axis=1)], axis=0)
        x1 = tf.reshape(x1, x1_shape)
        
        y1_shape = y.shape[1:]
        y1 = tf.concat([tf.concat([simg0, simg1], axis=1), 
                        tf.concat([simg2, simg3], axis=1)], axis=0)
        y1 = tf.reshape(y1, y1_shape)
        
        x_list.append(x1)
        y_list.append(y1)
        
    x, y = tf.stack(x_list, axis=0), tf.stack(y_list, axis=0)
    return x, y

def build_batch_augment():
    setting_cfg = get_settings('setting.yaml')
    aug_cfg = get_settings('augment.yaml')

    im_size = setting_cfg['im_size']
    batch_size = setting_cfg['BATCH_SIZE']

    def augment_batch_image_mask(images, masks):
        images, masks = batch_mixup_seg(images, masks, batch_size, aug_cfg['mixup_prob'])
        images, masks = batch_cutmix_seg(images, masks, im_size, batch_size, aug_cfg['cutmix_prob'])

        P = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['mosaic_prob'], tf.int32)
        if P == 1:
            images, masks = batch_mosaic_seg(images, masks, im_size, im_size, batch_size, aug_cfg['mosaic_range'])
        else:
            P = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['mosaic_full_prob'], tf.int32)
            if P == 1:
                images, masks = batch_mosaic_full_seg(images, masks, batch_size, aug_cfg['mosaic_full_range'])

        return images, masks

    return augment_batch_image_mask

if __name__ == '__main__':
    import cv2
    
    settings = get_settings()
    globals().update(settings)

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    image_path = '/storage/hieunmt/tf_segment/unzip/polyp/TrainDataset/images/1.png'
    mask_path = '/storage/hieunmt/tf_segment/unzip/polyp/TrainDataset/masks/1.png'

    img = cv2.imread(image_path)
    img = cv2.resize(img, (im_size, im_size))
    cv2.imwrite("sample_aa_img.png", img)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (im_size, im_size))
    mask = np.expand_dims(mask, -1)
    cv2.imwrite("sample_aa_mask.png", mask)

    img = img[...,::-1]
    img = img / 255.0
    img = np.float32(img)

    mask = mask[...,::-1]
    mask = mask / 255.0
    mask = np.float32(mask)

    augment_img = build_augment()
    img, mask = augment_img(img, mask)

    # # img = rotate_image(img, rotate)

    # # img = cutout(img, cutout_pad_factor)

    # # img = color_image(img, hue, sature, contrast, brightness)

    # # img = solarize(img, solarize_threshold)

    # # img = equalize_image(img)

    # img = clip_image(img)

    img = img.numpy()[...,::-1] * 255
    cv2.imwrite("sample_aug_img.png", img)
    mask = mask.numpy()[...,::-1] * 255
    cv2.imwrite("sample_aug_mask.png", mask)