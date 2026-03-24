#!/usr/bin/env python3

import json
import os
import numpy as np
import random
import keras

def find_next_valid_index(current_label, used_mask):
    for i in range(len(y_test)):
        if not used_mask[i] and y_test[i] != current_label:
            return i
    return None

if __name__ == '__main__':
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    input_nn = 20

    test_data_1 = np.load("../../haptic_data/data1/global_normalized_test_data_500ms.npy")
    x_test = test_data_1[:, :-1]
    y_test = test_data_1[:, -1]
    x_test = np.reshape(x_test, (test_data_1.shape[0], 50, 13, 1))
    x_test = x_test[:, :20, 1:, :]

    # TESTING DATASET1 to EXCLUDE "BAD" SAMPLES ------------------------------------------------------------------------
    cnn_model = "v1_1"
    lstm_model = "v1_2"
    transformer_model = "v1_1"
    model_cnn = keras.models.load_model("../../convolutional/v1/"+cnn_model+"/cnn_"+cnn_model+".keras")
    model_lstm = keras.models.load_model("../../recurrent/v1/"+lstm_model+"/lstm_"+lstm_model+".keras")
    model_transformer = keras.models.load_model("../../transformers/v1/"+transformer_model+"/transformer_"+transformer_model+".keras")
    rows_to_remove = []
    for i in range(0, len(test_data_1)):
        prediction_cnn = model_cnn.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
        prediction_lstm = model_lstm.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
        prediction_transformer = model_transformer.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
        decoded_prediction_cnn = np.argmax(prediction_cnn, axis=1, out=None)
        decoded_prediction_lstm = np.argmax(prediction_lstm, axis=1, out=None)
        decoded_prediction_transformer = np.argmax(prediction_transformer, axis=1, out=None)
        true = y_test[i]
        if true != decoded_prediction_cnn or true != decoded_prediction_lstm or true != decoded_prediction_transformer:
            rows_to_remove.append(i)
    print("len(rows_to_remove):")
    print(len(rows_to_remove))
    filtered_data = np.delete(test_data_1, rows_to_remove, axis=0)



    # CONCATENATING 7 ACTIONS AT A TIME (139 SAMPLES) ------------------------------------------------------------------------
    x_test = filtered_data[:, :-1]
    y_test = filtered_data[:, -1]
    x_test = np.reshape(x_test, (filtered_data.shape[0], 50, 13, 1))
    x_test = x_test[:, :, 1:, :]

    # Combine and shuffle to allow random sampling
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    used = np.zeros(len(x_test), dtype=bool)

    x_samples = []
    y_samples = []

    i = 0
    while len(x_samples) < 139:
        current_sequence = []
        current_labels = []
        last_label = -1

        for _ in range(7):
            found = False
            for j in range(len(y_test)):
                if not used[j] and y_test[j] != last_label:
                    current_sequence.append(x_test[j])
                    current_labels.append(y_test[j])
                    used[j] = True
                    last_label = y_test[j]
                    found = True
                    break
            if not found:
                # Not enough remaining samples that meet the condition — reshuffle and try again
                break

        if len(current_sequence) == 7:
            concatenated = np.concatenate(current_sequence, axis=0)  # shape (350, 12, 1)
            x_samples.append(concatenated)
            y_samples.append(current_labels)

    x_samples = np.array(x_samples)  # shape (139, 350, 12, 1)
    y_samples = np.array(y_samples)  # shape (139, 7)

    print("\nx_samples")
    print(type(x_samples))
    print(x_samples.shape)
    print("Y_samples")
    print(type(y_samples))
    print(y_samples.shape)

    np.save("x_test_data.npy", x_samples)
    np.save("y_test_data.npy", y_samples)
