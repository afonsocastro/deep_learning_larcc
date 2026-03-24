#!/usr/bin/env python3

import keras
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.utils import plot_confusion_matrix_percentage, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay
from progressbar import progressbar
from deep_learning_larcc.utils import NumpyArrayEncoder
import numpy as np
from deep_learning_larcc.config.definitions import ROOT_DIR

if __name__ == '__main__':
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    tresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    input_nn = 20
    test_data_1 = np.load(ROOT_DIR + "/haptic_data/data1/global_normalized_test_data_20ms.npy")
    x_test = test_data_1[:, :-1]
    y_test = test_data_1[:, -1]
    x_test = np.reshape(x_test, (test_data_1.shape[0], input_nn, 13, 1))
    x_test = x_test[:, :, 1:, :]
    cnn_model = "v1_1"
    lstm_model = "v1_2"
    transformer_model = "v1_1"

    model_cnn = keras.models.load_model("../convolutional/v1/"+cnn_model+"/cnn_"+cnn_model+".keras")
    model_lstm = keras.models.load_model("../recurrent/v1/"+lstm_model+"/lstm_"+lstm_model+".keras")
    model_transformer = keras.models.load_model("../transformers/v1/"+transformer_model+"/transformer_"+transformer_model+".keras")
    rejected_percentage = {"cnn": [], "lstm": [], "transformer": []}
    histogram_data = {"cnn": {"accuracy": [], "recall": [], "precision": [], "f1": []},
               "lstm": {"accuracy": [], "recall": [], "precision": [], "f1": []},
               "transformer": {"accuracy": [], "recall": [], "precision": [], "f1": []}}

    for model, model_name in zip([model_cnn, model_lstm, model_transformer], ["cnn", "lstm", "transformer"]):
        pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

        for treshold in tresholds:
            valid_count = 0
            # for i in range(0, len(test_data_1)):
            print("\n")
            print(
                "-------------------------------------------------------------------------------------------------")
            print(f"TESTING for {treshold} treshold, for {model_name} model")
            print("Loading ...")
            print(
                "-------------------------------------------------------------------------------------------------")
            print("\n")
            for i in progressbar(range(len(test_data_1)), redirect_stdout=True):
                prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)

                if np.max(prediction) >= treshold:
                    # Reverse to_categorical from keras utils
                    decoded_prediction = np.argmax(prediction, axis=1, out=None)

                    # true = test_data[i, -1]
                    true = y_test[i]

                    prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
                    prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
                    prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
                    prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)
                    valid_count += 1
                else:
                    continue

            rejected_percentage[model_name].append((treshold,((len(test_data_1)-valid_count)/len(test_data_1))*100))
            # -------------------------------------------------------------------------------------------------------------
            # METRICS-----------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------------
            metrics_pull = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_push = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_shake = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_twist = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

            simple_metrics_calc(pull, metrics_pull)
            simple_metrics_calc(push, metrics_push)
            simple_metrics_calc(shake, metrics_shake)
            simple_metrics_calc(twist, metrics_twist)

            for m in ["accuracy", "recall", "precision", "f1"]:
                histogram_data[model_name][m].append((treshold, (metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4))
    with open("metrics_histograms_treshold_100_dataset1.json", "w") as write_file:
        json.dump(histogram_data, write_file, cls=NumpyArrayEncoder)
    with open("rejected_samples_percentage_histograms_treshold_dataset1.json", "w") as write_file_rej:
        json.dump(rejected_percentage, write_file_rej, cls=NumpyArrayEncoder)

    #
    # # Extract x and y values for each dataset
    # x_cnn, y_cnn = zip(*histogram_data["cnn"]["accuracy"])
    # x_lstm, y_lstm = zip(*histogram_data["lstm"]["accuracy"])
    # x_transformer, y_transformer = zip(*histogram_data["transformer"]["accuracy"])
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(x_cnn, y_cnn, marker='o', label="CNN")
    # plt.plot(x_lstm, y_lstm, marker='s', label="LSTM")
    # plt.plot(x_transformer, y_transformer, marker='^', label="Transformer")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Plot with 3 Lines")
    # plt.legend()
    # plt.grid(True)
    # plt.show()