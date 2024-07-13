import os
from PIL import Image
import numpy as np
import time
from utilities import create_dir, load_data, save_data, is_img_empty, showimg
import sys
import cv2

# %%


def preprocess(images, new_width, new_height):
    new_images = np.array([cv2.resize(img,
                            (new_height, new_width), 
                            interpolation=cv2.INTER_AREA)
                            for img in images], dtype=np.float32)
    #new_images = np.array(images, dtype=np.float32)

    new_images /= 255
    new_images = np.reshape(new_images, (*new_images.shape, 1))
    new_images = 1 - new_images
    return new_images


# %%

start_time = time.time()

white = 'white'
np.set_printoptions(threshold=sys.maxsize)

base_dr = '.'

load_trial = False
trial = '_trial' if load_trial else ''

print('loading data...')

train_images, train_labels = load_data('train_bits' + trial)
val_images, val_labels = load_data('validation_bits' + trial)
test_images, test_labels = load_data('test_bits' + trial)

print('preprocessing...')
height, width = [x // 2 for x in train_images.shape[1:]]

train_images = preprocess(train_images, height, width)
val_images = preprocess(val_images, height, width)
test_images = preprocess(test_images, height, width)

print('saving images...')

save_data(train_images, train_labels, 'train_bits_processed' + trial)
save_data(val_images, val_labels, 'validation_bits_processed' + trial)
save_data(test_images, test_labels, 'test_bits_processed' + trial)

print('Duration: ', time.time() - start_time)
