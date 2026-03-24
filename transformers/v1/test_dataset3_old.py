#!/usr/bin/env python3

import keras
import numpy as np
from numpy import save
from progressbar import progressbar


if __name__ == '__main__':
    sliding_window = 20
    time_steps = 6000

    version = "v1_1"
    model_name = "transformer_" + version
    transformer_model = keras.models.load_model(version + "/transformer_" + version + ".keras")

    test_data = np.load("../../haptic_data/data3_old/global_normalized_data.npy")
    n_test = test_data.shape[0]
    x_test = np.reshape(test_data[:, :, :-1], (int(n_test), time_steps, 13, 1))
    x_test = x_test[:, :, 1:, :]
    y_test = test_data[:, :, -1]

    pred_transformer = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
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

    save(version + "/data3_pred_transformer_" + version + ".npy", pred_transformer)
    save('true_results_data3.npy', y_test)
