#!/usr/bin/env python3

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import load
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from progressbar import progressbar
from deep_learning_larcc.utils import NumpyArrayEncoder


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist


if __name__ == '__main__':
    time_steps = 100
    sliding_window = 20

    version = "v1_5"
    model_name = "lstm_" + version

    pred_lstm__data = load(version + "/data2_pred_lstm_" + version + ".npy")
    all_true = np.load("../../haptic_data/data2/y_test_data.npy")

    print(len(all_true))

    times = np.array([i for i in range(0, time_steps)])
    total_pull_seq2label, total_push_seq2label, total_shake_seq2label, total_twist_seq2label = 0, 0, 0, 0
    total_count_lstm_ = 0
    lstm__miss_classification_pos_transition = []
    lstm__previous_prediction_is_2nd_primitive = []
    n_samples_approved_reactivity = 0
    area_count = 0
    dict_react_count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                        "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0, "20": 0}
                        # "21": 0, "22": 0, "23": 0, "24": 0, "25": 0}
    for n in range(0, int(len(all_true))):

        old_true = all_true[n]
        true = np.zeros(100)
        for i in range(0, 50):
            true[i] = old_true[0]
        for i in range(50, 100):
            true[i] = old_true[1]

        pred_lstm_ = pred_lstm__data[n]
        pull_lstm_, push_lstm_, shake_lstm_, twist_lstm_ = value_for_array(pred_lstm_, time_steps - sliding_window + 1)
        count_lstm_ = 0
        lstm__correctly_predicted = False
        changed_true = False
        previous_prediction_is_2nd_primitive = False

        for i in times:
            if i > 18:
                confidences_i = [pull_lstm_[i - 19], push_lstm_[i - 19], shake_lstm_[i - 19], twist_lstm_[i - 19]]
                pred_lstm__i = np.array(confidences_i).argmax()

                # REACTIVITY-------------------------------------------------
                if true[i] != true[i - 1]:
                    changed_true = True
                    if old_prediction == true[i]:
                        previous_prediction_is_2nd_primitive = True
                if changed_true:
                    if pred_lstm__i == true[i]:
                        changed_true = False
                        lstm__correctly_predicted = True
                    else:
                        count_lstm_ += 1

                # AREA------------------------------------
                # if true[i] == pred_lstm__i:
                area_count += confidences_i[int(true[i])]
                old_prediction = pred_lstm__i

        if lstm__correctly_predicted and not previous_prediction_is_2nd_primitive:
            total_count_lstm_ += count_lstm_
            n_samples_approved_reactivity += 1
            for key, value in dict_react_count.items():
                if count_lstm_ == int(key):
                    dict_react_count[key] += 1
        elif not lstm__correctly_predicted:
            lstm__miss_classification_pos_transition.append({"idx": n, "gt": old_true})
        elif previous_prediction_is_2nd_primitive:
            lstm__previous_prediction_is_2nd_primitive.append({"idx": n, "gt": old_true})

    reactivity_metric = total_count_lstm_ / n_samples_approved_reactivity
    area_metric = np.array(area_count / ((time_steps-sliding_window)*len(all_true)))

    print("\ndict_react_count")
    print(dict_react_count)
    total_dict_count = 0
    for key, value in dict_react_count.items():
        total_dict_count += dict_react_count[key]

    print("\ntotal_dict_count:")
    print(total_dict_count)

    keys = sorted(dict_react_count.keys(), key=int)
    values = [dict_react_count[k] for k in keys]
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values, color='royalblue')
    plt.xlabel("Timesteps until change to correct prediction")
    plt.title("LSTM " + version +" (Mean Reactivity: "+ str(round(reactivity_metric,4)) +")")
    plt.show()

    print("\nLSTM " + str(version) + " Reactivity Metric:")
    print(reactivity_metric)

    print("\nlstm_ " + str(version) + " Pre-Transition Prediction being (wrongly) 2nd primitive:")
    print(lstm__previous_prediction_is_2nd_primitive)
    n_pp2p = len(lstm__previous_prediction_is_2nd_primitive)
    print(f"{n_pp2p} - {(n_pp2p / 565) * 100}%")

    print("\nlstm_ " + str(version) + " Miss Classification Pos Transition Metric:")
    print(lstm__miss_classification_pos_transition)
    n_mcpt = len(lstm__miss_classification_pos_transition)
    print(f"{n_mcpt} - {(n_mcpt/565)*100}%")

    print("\nlstm " + str(version) + " Area Metric:")
    print(area_metric)

    # reactivity_metric_dict = {"area": area_metric,
    #                           "reactivity": reactivity_metric,
    #                           "pos_transition_miss_classification": lstm__miss_classification_pos_transition,
    #                           "prev_predict_is_2nd_action": lstm__previous_prediction_is_2nd_primitive}
    #
    # with open("data2_metrics_" + model_name + ".json", "w") as write_file:
    #     json.dump(reactivity_metric_dict, write_file, cls=NumpyArrayEncoder)
