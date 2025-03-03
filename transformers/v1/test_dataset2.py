#!/usr/bin/env python3

import keras
from numpy import save
from progressbar import progressbar
import numpy as np


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    print("time_steps-sliding_window+1")
    print(time_steps-sliding_window+1)

    x_test = np.load("../../haptic_data/data2/x_test_data.npy")
    # n_test = x_test.shape[0]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_test = x_test[:, :, 1:, :]
    # y_test = np.load("../../haptic_data/data2/y_test_data.npy")

    version = "v1_4"
    model_name = "transformer_" + version
    transformer_model = keras.models.load_model(version + "/transformer_" + version + ".keras")

    # Transformer TESTING
    pred_transformer = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = transformer_model.predict(x=x_test[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_transformer.append(sample_pred)

    pred_transformer = np.array(pred_transformer)
    print("\n")
    print("pred_transformer.shape")
    print(pred_transformer.shape)
    print("\n")
    pred_transformer = np.reshape(pred_transformer, (pred_transformer.shape[0], pred_transformer.shape[1], pred_transformer.shape[3]))

    save(version + "/data2_pred_transformer_" + version + ".npy", pred_transformer)