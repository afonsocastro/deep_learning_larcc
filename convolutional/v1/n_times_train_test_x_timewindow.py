#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs: INFO, WARN, and DEBUG
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from sklearn.metrics import confusion_matrix
from deep_learning_larcc.utils import prediction_classification_absolute
from deep_learning_larcc.config.definitions import ROOT_DIR
import numpy as np
import json
from progressbar import progressbar
from deep_learning_larcc.utils import NumpyArrayEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # create model

def create_cnn_v1_3(nn):
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(nn, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_3"

def create_cnn_v1_1(nn):
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(13, 1), activation="relu", input_shape=(nn, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_1"

# def create_convolutional_nn_old(input):
#     # Create model
#     modelo = Sequential()
#     modelo.add(Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(input, 13, 1)))
#     modelo.add(MaxPooling2D((2, 1)))
#
#     modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
#     modelo.add(MaxPooling2D((2, 1)))
#
#     modelo.add(Flatten())
#     modelo.add(Dense(4, activation="softmax"))
#
#     # compile model using accuracy to measure model performance
#     modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return modelo, "cnn_v1_3"

if __name__ == '__main__':
    n_times = 100
    validation_split = 0.3
    # x_times = [10, 20, 30, 40, 50]
    # for x_time in x_times:
    x_time = 50
    training_test_list = []
    for n in progressbar(range(n_times), redirect_stdout=True):
        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print(f"TRAINING: {n} simulation, for {x_time} timesteps (window size)")
        print("Loading ...")
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        training_data = np.load(ROOT_DIR + "/haptic_data/data1/global_normalized_train_data_"+str(x_time)+"0ms.npy")
        x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], x_time, 13))
        y_train = to_categorical(training_data[:, -1])
        x_train = x_train[:, :, 1:]

        # model, model_name = create_cnn_v1_3(x_time)
        model, model_name = create_cnn_v1_1(x_time)
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        # model, model_name = create_convolutional_nn_old(x_time)
        model.summary()
        # epochs = 500
        fit_history = model.fit(x_train, y_train, shuffle=True, validation_split=validation_split, epochs=500,
                                batch_size=64)

        print("\n")
        print("Using %d samples for training and %d for validation" % (
        len(training_data) * (1 - validation_split), len(training_data) * validation_split))

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print(f"TESTING: {n} simulation, for {x_time} timesteps (window size)")
        print("Loading ...")
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        test_data_1 = np.load(ROOT_DIR + "/haptic_data/data1/global_normalized_test_data_"+str(x_time)+"0ms.npy")

        x_test = test_data_1[:, :-1]
        y_test = test_data_1[:, -1]
        x_test = np.reshape(x_test, (test_data_1.shape[0], x_time, 13, 1))

        x_test = x_test[:, :, 1:, :]

        predictions_list = []

        pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

        for i in range(0, len(test_data_1)):
            prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
            # Reverse to_categorical from keras utils
            decoded_prediction = np.argmax(prediction, axis=1, out=None)
            true = y_test[i]

            prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
            prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
            prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
            prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

            predictions_list.append(decoded_prediction)

        predicted_values = np.asarray(predictions_list)

        cm = confusion_matrix(y_true=y_test, y_pred=predicted_values)
        cm_true = cm / cm.astype(float).sum(axis=1)
        cm_true_percentage = cm_true * 100

        test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
        training_test_dict = {"training": fit_history.history, "test": test_dict}
        training_test_list.append(training_test_dict)

    with open(str(n_times) + "_times_train_test_" + model_name + "_time_window_"+str(x_time)+ "_ts.json", "w") as write_file:
        json.dump(training_test_list, write_file, cls=NumpyArrayEncoder)
