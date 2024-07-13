from scipy.io import loadmat
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import shutil

def create_dir(dr):
    if os.path.exists(dr):
        shutil.rmtree(dr)
    os.mkdir(dr)
    return dr

def save_data(images, labels, name):
    
    np.savez_compressed(name + '.npz',
                     images, labels)

    print(name, 'data saved to disk')
    
def load_data(name):
    dict_data = np.load(name + '.npz', allow_pickle=True)

    images = dict_data['arr_0']
    labels = dict_data['arr_1']

    print(name, 'data loaded from disk')
    
    return images, labels
    
def parseMatData(data_file_name):
    stam = loadmat(data_file_name) # peak_file is the file name
    peaks_indices = stam['peaks_indices'].flatten()
    index_of_max_in_peak_indices = stam['index_of_max_in_peak_indices'].flatten()[0]
    SCALE_FACTOR = stam['SCALE_FACTOR'].flatten()[0]
    delta = stam['delta'].flatten()[0]
    top_test_area = stam['top_test_area'].flatten()[0]
    bottom_test_area = stam['bottom_test_area'].flatten()[0]
    return peaks_indices, index_of_max_in_peak_indices, SCALE_FACTOR, delta, top_test_area, bottom_test_area

def save_model(model, name='model'):
    # Serialize model to JSON
    model_json = model.to_json()

    # Save model architecture to JSON
    with open(f"{name}.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(f"{name}.h5")

    print("Saved model to disk")
    

def load_model(name='model'):
    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '.h5')
    print('Loaded model from disk')
    return loaded_model
    

def show_plots(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def showimg(img):
    fig, axs = plt.subplots(figsize=(6, 6))

    axs.imshow(img, cmap='gray')
    axs.axis('off')

    plt.show()

def is_row_color(img_row, color, error_rate):
    num_of_fitting = 0
    for pixel in img_row:
        num_of_fitting += (pixel == color)

    return num_of_fitting / img_row.size > (1 - error_rate)


def is_row_color_rigid(img_row, color):
    for pixel in img_row:
        if pixel != color:
            return False
    return True


def is_img_empty(img, num_of_rows_to_check, max_empty_rows, error_rate):
    white_value = 255

    img_pixels = np.array(img)
    rows = img_pixels.shape[0]
    rows_to_check = [i * rows // (num_of_rows_to_check + 1) for i in range(1, num_of_rows_to_check + 1)]
    empty_rows = 0
    for row in rows_to_check:
        if is_row_color(img_pixels[row], white_value, error_rate):
            empty_rows += 1
    return empty_rows >= max_empty_rows
