import os
from PIL import Image
import numpy as np
import time
from utilities import create_dir, load_data, save_data, is_img_empty
import sys

# %%

def cut_line_to_bits(line_img, bit_width, start=0, congruence=0):
    height, width = line_img.shape
    bits = []
    for bit_start in range(start, width - bit_width + 1, (bit_width - congruence)):
        bits.append(line_img[0:height, bit_start:bit_start + bit_width])

    return bits


def cut_data_to_bits(images, labels, bit_width, start_positions, save_dr='', removed_dr=''):
    bits_rows_to_check = 7
    bits_max_empty_rows = 6
    save_images = os.path.exists(save_dr)
    save_removed_bits = os.path.exists(removed_dr)
    bits_of_images = []
    new_labels = []
    bits_index = 0
    removed_index = 0
    error_rate = 0.05
    for img, label in zip(images, labels):
        for start in start_positions:
            bits = cut_line_to_bits(img, bit_width, start)
            for bit in bits:
                if save_images:
                    bit_img = Image.fromarray(bit, mode="L")
                if not is_img_empty(bit, bits_rows_to_check, bits_max_empty_rows, error_rate) and not \
                        is_img_empty(np.transpose(bit), bits_rows_to_check, bits_max_empty_rows, error_rate):
                    bits_of_images.append(bit)
                    new_labels.append(label)
                    if save_images:
                        name = str(bits_index)
                        bit_img.save(os.path.join(save_dr, name + ".jpg"))
                        bits_index += 1

                elif save_removed_bits:
                    name = str(removed_index)
                    bit_img.save(os.path.join(removed_dr, name + ".jpg"))
                    removed_index += 1

    return bits_of_images, new_labels
    

# %%

start_time = time.time()

white = 'white'
np.set_printoptions(threshold=sys.maxsize)

bit_width = 205

save_images = False

base_dr = '.'

training_bits_dr = create_dir(os.path.join(base_dr, 'training_bits')) if save_images else ''
validation_bits_dr = create_dir(os.path.join(base_dr, 'valid_bits')) if save_images else ''
test_bits_dr = create_dir(os.path.join(base_dr, 'test_bits')) if save_images else ''
removed_bits_dr = create_dir(os.path.join(base_dr, 'removed_bits')) if save_images else ''

load_trial = False
trial = '_trial' if load_trial else ''

print('loading data...')

orig_train_data, orig_train_labels = load_data('train_spiltted' + trial)
orig_val_data, orig_val_labels = load_data('validation' + trial)
orig_test_data, orig_test_labels = load_data('test' + trial)

print('cutting data to bits...')

print('cutting train...')
start_list = range(0, 161, 40)
train_data, train_labels = cut_data_to_bits(orig_train_data, orig_train_labels, bit_width, start_list,
                                            training_bits_dr, removed_bits_dr)

print('cutting val...')
val_data, val_labels = cut_data_to_bits(orig_val_data, orig_val_labels, bit_width, start_list,
                                        validation_bits_dr, removed_bits_dr)

print('cutting test...')
test_data, test_labels = cut_data_to_bits(orig_test_data, orig_test_labels, bit_width, start_list, 
                                          test_bits_dr, removed_bits_dr)


train_data, train_labels = np.array(train_data), np.array(train_labels)
val_data, val_labels = np.array(val_data), np.array(val_labels)
test_data, test_labels = np.array(test_data), np.array(test_labels)

print('saving data...')

save_data(train_data, train_labels, 'train_bits' + trial)
save_data(val_data, val_labels, 'validation_bits' + trial)
save_data(test_data, test_labels, 'test_bits' + trial)

print('Duration: ', time.time() - start_time)
