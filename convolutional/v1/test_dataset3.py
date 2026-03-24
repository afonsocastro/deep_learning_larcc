#!/usr/bin/env python3

import keras
from numpy import save
from progressbar import progressbar
import numpy as np


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 350
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    x_test = np.load("../../haptic_data/data3/x_test_data.npy")
    version = "v1_1"
    model_name = "cnn_" + version
    cnn_model = keras.models.load_model(version + "/cnn_" + version + ".keras")

    # cnn TESTING
    pred_cnn = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = cnn_model.predict(x=x_test[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_cnn.append(sample_pred)

    pred_cnn = np.array(pred_cnn)
    print("\n")
    print("pred_cnn.shape")
    print(pred_cnn.shape)
    print("\n")
    pred_cnn = np.reshape(pred_cnn, (pred_cnn.shape[0], pred_cnn.shape[1], pred_cnn.shape[3]))

    save(version + "/data3_pred_cnn_" + version + ".npy", pred_cnn)