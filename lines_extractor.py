import os
from PIL import Image
import numpy as np
import time
from utilities import create_dir, save_data, parseMatData, is_img_empty
import math

# %%

base_dr = '.'
data_dr = '.\data'
info_dr = os.path.join(data_dr, '5_DataDarkLines')
image_dr = os.path.join(data_dr, '3_ImagesLinesRemovedBW')

start_time = time.time()

save_images = False

# save images as files
if save_images:
    training_dr = create_dir(os.path.join(base_dr, 'training'))
    test_dr = create_dir(os.path.join(base_dr, 'test'))
    removed_dr = create_dir(os.path.join(base_dr, 'removed'))

X = []
y = []
test_images = []
test_labels = []
X_trial = []
y_trial = []
test_images_trial = []
test_labels_trial = []

num_of_files = len(os.listdir(info_dr))
num_of_files = 200
num_of_dev = math.ceil(num_of_files / 2)
num_of_trial = num_of_files - num_of_dev

line_final_height = 205

current_X = X
current_y = y
current_test_images = test_images
current_test_labels = test_labels

removed_index = 0
removed_bits_index = 0

# variables for checking if image is empty
error_rate = 0.01
num_of_rows_to_check = 3
max_empty_rows = 2

# %%

to_sub = 0

print('extracting lines...')
# run on all files in data folder and extract lines and test areas from each
for index, file_name in enumerate(os.listdir(info_dr)):
    if index == num_of_files:
        break

    # switch save to trial
    if index == num_of_dev:
        print('switch to trial')
        current_X = X_trial
        current_y = y_trial
        current_test_images = test_images_trial
        current_test_labels = test_labels_trial
        to_sub = num_of_dev

    print(file_name)

    file_name = os.path.splitext(file_name)[0]
    data_file_name = os.path.join(info_dr, file_name)

    peaks_indices, index_of_max_in_peak_indices, SCALE_FACTOR, delta, \
    top_test_area, bottom_test_area = parseMatData(data_file_name)

    peaks_indices *= SCALE_FACTOR

    img_file_name = os.path.join(image_dr, file_name) + '.jpg'

    img = Image.open(img_file_name)
    width, height = img.size
    # extract all lines and test area from the image
    full_line_index = 0
    for i in range(peaks_indices.size - 1):
        isTest = False
        top = peaks_indices[i]
        bottom = peaks_indices[i + 1]
        # check if current line is test
        if top < top_test_area < bottom:
            top = top_test_area
            peaks_indices[i + 1] = bottom_test_area
            isTest = True

        line_height = bottom - top
        line_img = img.crop((0, top, width, bottom))

        # if line is not empty or is the test line, save
        if isTest or not is_img_empty(line_img, num_of_rows_to_check, max_empty_rows, error_rate):
            full_line_index += 0 if isTest else 1
            name = str(index) if isTest else str(index) + '_' + str(full_line_index)
            result = Image.new(img.mode, (width, line_final_height), 'white')
            result.paste(line_img, (0, line_final_height - line_height))
            result_array = np.asarray(result, dtype=np.uint8)
            if isTest:
                current_test_images.append(result_array)
                current_test_labels.append(index - to_sub)
                if save_images:
                    line_img.save(os.path.join(test_dr, name + ".jpg"))

            else:
                current_X.append(result_array)
                current_y.append(index - to_sub)
                if save_images:
                    line_img.save(os.path.join(training_dr, name + ".jpg"))

# %%

print('converting to numpy arrays...')

X = np.asarray(X, dtype=np.uint8)
y = np.asarray(y, dtype=np.uint16)
test_images = np.asarray(test_images, dtype=np.uint8)
test_labels = np.asarray(test_labels, dtype=np.uint16)
X_trial = np.asarray(X_trial, dtype=np.uint8)
y_trial = np.asarray(y_trial, dtype=np.uint16)
test_images_trial = np.asarray(test_images_trial, dtype=np.uint8)
test_labels_trial = np.asarray(test_labels_trial, dtype=np.uint16)

# %%

print('saving data...')

save_data(X, y, 'train')
save_data(test_images, test_labels, 'test')
save_data(X_trial, y_trial, 'train_trial')
save_data(test_images_trial, test_labels_trial, 'test_trial')

print('Duration: ', time.time() - start_time)
