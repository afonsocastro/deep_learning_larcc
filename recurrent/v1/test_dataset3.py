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

    version = "v1_2"
    model_name = "lstm_" + version
    lstm_model = keras.models.load_model(version + "/lstm_" + version + ".keras")

    # LSTM TESTING
    pred_lstm = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = lstm_model.predict(x=x_test[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_lstm.append(sample_pred)

    pred_lstm = np.array(pred_lstm)
    print("\n")
    print("pred_lstm.shape")
    print(pred_lstm.shape)
    print("\n")
    pred_lstm = np.reshape(pred_lstm, (pred_lstm.shape[0], pred_lstm.shape[1], pred_lstm.shape[3]))

    save(version + "/data3_pred_lstm_" + version + ".npy", pred_lstm)