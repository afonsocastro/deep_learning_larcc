#!/usr/bin/env python3

from keras.models import load_model
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder
from numpy import save
from progressbar import progressbar
import numpy as np
from config.definitions import ROOT_DIR


if __name__ == '__main__':

    sliding_window = 20
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']

    x_test = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data.npy")
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_test = x_test[:, :, :-1, :]

    # version = "v1_2"
    # model = load_model(ROOT_DIR + "/recurrent/models/"+version+"/lstm_"+version+".keras")
    version = "v1_1"
    # model = load_model(ROOT_DIR + "/convolutional/models/" + version + "/cnn_" + version + ".keras")
    model = load_model(ROOT_DIR + "/transformers/models/" + version + "/transformer_" + version + ".keras",
        custom_objects={"SinePositionEncoding": SinePositionEncoding, "TransformerEncoder": TransformerEncoder},compile=False)

    # TESTING
    pred = []

    for i in progressbar(range(len(x_test))):
        sample = x_test[i]
        windows = np.array([sample[i:i + sliding_window] for i in range(sample.shape[0] - sliding_window + 1)])

        sample_pred = model.predict(windows, batch_size=64, verbose=0)
        pred.append(sample_pred)

    pred = np.array(pred)

    print("\n")
    print("pred.shape")
    print(pred.shape)
    print("\n")
    save("data3_pred_transformer_" + version + ".npy", pred)