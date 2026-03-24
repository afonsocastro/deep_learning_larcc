#!/usr/bin/env python3

import keras
import json
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.utils import plot_confusion_matrix_percentage, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay
from progressbar import progressbar
from deep_learning_larcc.utils import NumpyArrayEncoderNumPy
import numpy as np
from deep_learning_larcc.config.definitions import ROOT_DIR

if __name__ == '__main__':
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    input_nn = 20
    test_data_1 = np.load(ROOT_DIR + "/haptic_data/data1/global_normalized_test_data_20ms.npy")
    epsilon = 1e-1
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
    entropy_dict = {"cnn": {"mean": 0, "std_dev": 0, "95_confidence_interval":0},
                    "lstm": {"mean": 0, "std_dev": 0, "95_confidence_interval":0},
                    "transformer": {"mean": 0, "std_dev": 0, "95_confidence_interval":0}}

    # gap_dict = {"cnn": {"mean": 0, "std_dev": 0, "95_confidence_interval": 0},
    #                 "lstm": {"mean": 0, "std_dev": 0, "95_confidence_interval": 0},
    #                 "transformer": {"mean": 0, "std_dev": 0, "95_confidence_interval": 0}}

    for model, model_name in zip([model_cnn, model_lstm, model_transformer], ["cnn", "lstm", "transformer"]):
        entropy_list = []
        gap_list = []
        for i in progressbar(range(len(test_data_1)), redirect_stdout=True):
            prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
            true = y_test[i]
            arr_safe = np.where(prediction < epsilon, epsilon, prediction)
            entropy = - np.sum(arr_safe * np.log2(arr_safe))
            # entropy = - np.sum(prediction * np.log2(prediction + 1e-8))
            sorted_arr = np.sort(prediction)  # Sorts in ascending order
            gap = sorted_arr[0][-1] - sorted_arr[0][-2]
            entropy_list.append(entropy)
            gap_list.append(gap)

        mean_entropy = np.mean(entropy_list)
        std_dev_entropy = np.std(entropy_list, ddof=1)  # Sample standard deviation
        ci_entropy = stats.t.interval(0.95, len(entropy_list) - 1, loc=mean_entropy, scale=std_dev_entropy / np.sqrt(len(entropy_list)))
        entropy_dict[model_name]["mean"] = round(mean_entropy, 5)
        entropy_dict[model_name]["std_dev"] = round(std_dev_entropy, 5)
        entropy_dict[model_name]["95_confidence_interval"] = [round(ci_entropy[0], 5), round(ci_entropy[1], 5)]

        # mean_gap = np.mean(gap_list)
        # std_dev_gap = np.std(gap_list, ddof=1)  # Sample standard deviation
        # ci_gap = stats.t.interval(0.95, len(gap_list) - 1, loc=mean_gap,
        #                       scale=std_dev_gap / np.sqrt(len(gap_list)))
        # gap_dict[model_name]["mean"] = round(mean_gap, 5)
        # gap_dict[model_name]["std_dev"] = round(std_dev_gap, 5)
        # gap_dict[model_name]["95_confidence_interval"] = [round(ci_gap[0], 5), round(ci_gap[1], 5)]

    with open("entropy_dataset1_1e1.json", "w") as write_file:
        json.dump(entropy_dict, write_file, cls=NumpyArrayEncoderNumPy)
    # with open("gap_analysis_dataset1.json", "w") as write_file_gap:
    #     json.dump(gap_dict, write_file_gap, cls=NumpyArrayEncoderNumPy)