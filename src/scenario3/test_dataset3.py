#!/usr/bin/env python3

from keras.models import load_model
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder
from numpy import save
from progressbar import progressbar
import numpy as np
from config.definitions import ROOT_DIR


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 1500
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    x_test = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_test = x_test[:, :, :-1, :]
    y_test = x_test[:, :, -1, :]

    version = "v1_2"
    model = load_model(ROOT_DIR + "/recurrent/models/"+version+"/lstm_"+version+".keras")

    # model = load_model(ROOT_DIR + "/transformers/models/" + version + "/transformer_" + version + ".keras",
    #     custom_objects={"SinePositionEncoding": SinePositionEncoding, "TransformerEncoder": TransformerEncoder},compile=False)

    # TESTING
    pred = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = model.predict(x=x_test[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred.append(sample_pred)

    pred = np.array(pred)
    print("\n")
    print("pred.shape")
    print(pred.shape)
    print("\n")
    pred = np.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[3]))

    save("data3_pred_lstm_" + version + ".npy", pred)