import numpy as np
import os
import time
from tensorflow.keras import models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import matplotlib.pyplot as plt
from PIL import Image
from utilities import save_model, load_model, load_data, showimg, save_data, show_plots
from keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.callbacks import BackupAndRestore, Callback
import sys
from sklearn.metrics import confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %% funcs

def split_images_by_label(images, labels):
    labels_set = list(set(labels))
    labels_set.sort()
    labels_count = [list(labels).count(label) for label in labels_set]
    start = 0
    images_by_label = []
    for label in labels_set:
        images_by_label.append(images[start:start + labels_count[label]])
        start += labels_count[label]

    return images_by_label

def predict_images(model, images):
    prediction = model.predict(images, verbose=0)
    max_label = prediction.shape[-1]
    votes = np.argmax(prediction, axis=-1)
    votes_count = [list(votes).count(vote) for vote in range(max_label)]
    chosen_label = votes_count.index(max(votes_count))
    return chosen_label

def evaluate_model(model, images, labels):
    labels = np.argmax(labels, axis=-1)  # to non categorical
    labels_set = list(set(labels))
    labels_set.sort()

    images_by_label = split_images_by_label(images, labels)

    predictions = []
    for label in labels_set:
        predicted_label = predict_images(model, images_by_label[label])
        predictions.append((label, predicted_label))

    accuracy = len(list(filter(lambda x: x[0] == x[1], predictions))) / len(labels_set)
    return accuracy, predictions

class EvaluateValidation(Callback):
    def __init__(self, val_images, val_labels):
        super(EvaluateValidation, self).__init__()
        self.val_images = val_images
        self.val_labels = val_labels
        
    def on_epoch_end (self, epochs, logs=None):
        
        # if epochs > 1:
            if logs == None: 
                return
            train_history['loss'].append(logs['loss'])
            train_history['accuracy'].append(logs['accuracy'])
              
            # Evaluate the model and get the validation accuracy
            val_accuracy = evaluate_model(self.model, self.val_images, self.val_labels)[0]
            
            # Append validation metrics to val_history
            val_history['val_loss'].append(logs['val_loss'])
            val_history['val_accuracy'].append(val_accuracy)
            
            print("val line accuracy:", evaluate_model(self.model, self.val_images, self.val_labels)[0])

            # save_data(np.array(train_history), np.array(val_history), "train val history")
            
            save_model(model, 'model ' + str(val_accuracy))
            save_data(np.array(train_history), np.array(val_history), "train val history "+ str(val_accuracy))
            print("saved model and history")


def preprocess(images, new_width, new_height):
    new_images = np.array([cv2.resize(img,#cv2.bilateralFilter(img, 9, 75, 75), 
                            (new_height, new_width), interpolation=cv2.INTER_AREA)
                            for img in images], dtype=np.float32)
    #new_images = np.array([img for img in images], dtype=np.float32)

    new_images /= 255
    new_images = np.reshape(new_images, (*new_images.shape, 1))
    new_images = 1 - new_images
    return new_images

# %%
np.set_printoptions(precision=2, suppress=True, threshold=sys.maxsize)

load_trial = False
trial = '_trial' if load_trial else ''

load_preprocessed = False

preprocessed = "_processed" if load_preprocessed else ''
    
print('loading data...')
train_images, train_labels = load_data('train_bits' + preprocessed + trial)
val_images, val_labels = load_data('validation_bits' + preprocessed + trial)
test_images, test_labels = load_data('test_bits' + preprocessed + trial)

# %%

height, width = [x // 2 for x in train_images.shape[1:]]

print('converting to one hot encoding...')
vec_size = max(train_labels) + 1
print("vec_size", vec_size)

if not load_preprocessed:
    print('preprocessing...')
    train_images= preprocess(train_images, height, width)
    val_images = preprocess(val_images, height, width)
    test_images = preprocess(test_images, height, width)
 
train_labels = to_categorical(train_labels, vec_size)
val_labels = to_categorical(val_labels, vec_size)
test_labels = to_categorical(test_labels, vec_size)

# %%
data_augmentation = ImageDataGenerator(rotation_range=20)

# %%
train_history = {'loss': [], 'accuracy': []}
val_history = {'val_loss': [], 'val_accuracy': []}

# %%
print('building model...')
model = models.Sequential()

# %%

model.add(Conv2D(filters=32, kernel_size=(7, 7), strides=(2,2), padding='same', activation='relu', input_shape=(height, width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Flatten())
model.add(Dropout(0.50))
model.add(Dense(units=vec_size*16, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(units=vec_size*8  , activation="relu"))
model.add(Dropout(0.50)) 
model.add(Dense(units=vec_size, activation='softmax'))

model.compile(optimizer=optimizers.Adam(learning_rate = 0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


print(model.summary())

# %%
print('fitting model...')
start_time = time.time()
flow = data_augmentation.flow(train_images, train_labels, batch_size=64, shuffle=True)
del train_images, train_labels
history = model.fit(flow, callbacks=[BackupAndRestore(backup_dir='.'), 
                    EvaluateValidation(val_images, val_labels)],
                    epochs=19, validation_data=(val_images, val_labels))

# %%
end_time = time.time()
print("time:", end_time - start_time)
show_plots(history.history)

#%%
# save end result
save_model(model, 'model')

#%%

print('evaluating model...')
score, predictions = evaluate_model(model, test_images, test_labels)  # model.evaluate(test_images, test_labels, verbose=0)
print("Test:", score)

wrong_predictions = list(filter(lambda x: x[0] != x[1], predictions))

#%%
true_labels, predicted_labels = list(zip(*wrong_predictions))

test_line_images, test_line_labels = load_data('test' + trial)

test_dict = dict(zip(test_line_labels, test_line_images))

wrong_test_lines = [(Image.fromarray(test_dict[correct_label], mode="L"), 
                       Image.fromarray(test_dict[predicted_label], mode="L"))
                             for correct_label, predicted_label
                             in wrong_predictions]

for correct_line_img, predicted_line_img in wrong_test_lines:
    showimg(correct_line_img)
    showimg(predicted_line_img)

# %%

true_labels, predicted_labels = list(zip(*predictions))
con = confusion_matrix(true_labels, predicted_labels, labels=range(vec_size))
plt.imshow(con, interpolation="None")

# %% evaluate from loaded

do_load_model = False
# do_load_model = True
if do_load_model:
    model = load_model("model")
    train_hist, val_hist = load_data("train val history")
    train_hist = train_hist.tolist()
    val_hist = val_hist.tolist()

    train_hist.update(val_hist)
    # new_hist = {}
    # new_hist.history=train_hist
    del val_hist
    # print(train_hist)

    show_plots(train_hist)
    
    print('evaluating model...')
    score, predictions = evaluate_model(model, test_images, test_labels)
    print("Test:", score)
    
    wrong_predictions = list(filter(lambda x: x[0] != x[1], predictions))
    
    true_labels, predicted_labels = list(zip(*wrong_predictions))
    
    test_line_images, test_line_labels = load_data('test' + trial)
    
    test_dict = dict(zip(test_line_labels, test_line_images))
    
    wrong_test_lines = [(Image.fromarray(test_dict[correct_label], mode="L"), 
                           Image.fromarray(test_dict[predicted_label], mode="L"))
                                 for correct_label, predicted_label
                                 in wrong_predictions]
    
    for correct_line_img, predicted_line_img in wrong_test_lines:
        showimg(correct_line_img)
        showimg(predicted_line_img)    
    
    true_labels, predicted_labels = list(zip(*predictions))
    con = confusion_matrix(true_labels, predicted_labels, labels=range(vec_size))
    plt.imshow(con, interpolation="None")
