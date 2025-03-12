#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # create model

# compile model using accuracy to measure model performance
# modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def create_cnn_v1_0():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(20, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(Conv2D(32, kernel_size=(1, 1), activation="relu"))
    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_0"

def create_cnn_v1_1():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(13, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_1"

def create_cnn_v1_2():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(9, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(3, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_2"

def create_cnn_v1_3():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(50, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_3"