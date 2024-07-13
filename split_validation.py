import numpy as np
import os
import time
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from utilities import load_data, save_data

# %%

start_time = time.time()

load_trial = False
trial = '_trial' if load_trial else ''

print('loading data...')
orig_train_data, orig_train_labels = load_data('train' + trial)

# %%

print('preparing...')
files_by_label = {}

labels = list(set(orig_train_labels))
labels.sort()

labels_count = [list(orig_train_labels).count(label) for label in labels]
val_nums = [1] * len(labels)
total_val = sum(val_nums)
total_train = orig_train_labels.shape[0] - total_val

height, width = orig_train_data.shape[1:]

train_data = np.zeros((total_train, height, width), dtype=np.uint8)
train_labels = np.zeros(total_train, dtype=np.uint16)
val_data = np.zeros((total_val, height, width), dtype=np.uint8)
val_labels = np.zeros(total_val, dtype=np.uint16)

# %%

i_train = 0
i_val = 0

print('splitting to train and validation...')
index = 0
for label in labels:
    print(label)
    lines_images = orig_train_data[index:index + labels_count[label]]
    np.random.shuffle(lines_images)
    index += labels_count[label]
    num_of_val = val_nums[label]
    for line in lines_images[num_of_val:]:
        train_data[i_train] = line
        train_labels[i_train] = label
        i_train += 1

    for line in lines_images[:num_of_val]:
        val_data[i_val] = line
        val_labels[i_val] = label
        i_val += 1

print('saving data...')

save_data(train_data, train_labels, 'train_spiltted' + trial)
save_data(val_data, val_labels, 'validation' + trial)

print('Duration: ', time.time() - start_time)
