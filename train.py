import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from keras.models import Sequential
import tensorflow as tf
import os
from keras.layers import Dropout # Used for deep NN

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    # model = Sequential([
    #     SqueezeNet(input_shape=(227, 227, 3), include_top=False),
    #     Dropout(0.5),
    #     Convolution2D(6, (1, 1), padding='valid'),
    #     Activation('relu'),
    #     MaxPooling2D((2,2))
    #     Convolution2D(4, (1, 1), padding='valid'),
    #     Activation('relu'),
    #     GlobalAveragePooling2D(),
    #     Activation('softmax')
    # ])
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    return model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper'],
    ...
]
'''
data, labels = zip(*dataset)
labels = list(map(mapper, labels))


'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels)

# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=5)

# save the model for later use
model.save("rock-paper-scissors-model.h5")
