#!/usr/bin/env python3

from numpy import load
import numpy as np
import pandas as pd
import json
from deep_learning_larcc.utils import NumpyArrayEncoder

def calculate_means(ts, true_values, predicti):
    values = {0: [0,0], 1: [0,0], 2: [0,0], 3: [0,0]}
    for i in ts:
        tv = int(true_values[i])
        values[tv][0] += predicti[i-19][tv]
        values[tv][1] += 1

    fractions = [(values[0][0], values[0][1]), (values[1][0], values[1][1]), (values[2][0], values[2][1]), (values[3][0], values[3][1])]

    valid_fractions = [num / den for num, den in fractions if den != 0]
    separated_mean= sum(valid_fractions) / len(valid_fractions)

    harmonic_fractions = [den / num for num, den in fractions if den != 0]
    h_mean = sum(harmonic_fractions) / len(valid_fractions)
    harmonic_mean = 1/h_mean

    valid_numerators = [num for num, den in fractions if den != 0]
    valid_denominators = [den for _, den in fractions if den != 0]
    global_mean = sum(valid_numerators) / sum(valid_denominators)
    return float(separated_mean), float(global_mean), float(harmonic_mean)

# def calculate_reactivity(ts, true_values, predicti):
#     values = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
#
#     count_= 0
#     correctly_predicted = False
#     changed_true = False
#     previous_prediction_is_2nd_primitive = False
#     for i in ts:
#         pred_i = np.array([predicti[i - 19][0], predicti[i - 19][1],predicti[i - 19][2], predicti[i - 19][3]]).argmax()
#         tv = int(true_values[i])
#         values[tv][0] += predicti[i-19][tv]
#         values[tv][1] += 1
#
#         if true_values[i] != true_values[i - 1]:
#             changed_true = True
#             if old_prediction == true_values[i]:
#                 previous_prediction_is_2nd_primitive = True
#         if changed_true:
#             if pred_i == true_values[i]:
#                 changed_true = False
#                 correctly_predicted = True
#             else:
#                 count_ += 1
#         old_prediction = pred_i
#     if correctly_predicted and not previous_prediction_is_2nd_primitive:
#         n_samples_approved_reactivity += 1

if __name__ == '__main__':
    time_steps = 6000
    sliding_window = 20
    models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models = ["cnn", "lstm", "transformer"]
    predictions = {}
    for model, version in zip(models, models_versions):
        predictions[model] = load("dataset3_old_results/data3_pred_" + model + version + ".npy")

    data = np.load("../haptic_data/data3_old/global_normalized_data.npy")
    y_labels = data[:, :, -1]
    times = np.array([i for i in range(19, time_steps)])
    mean_data = []
    total_data_count = {"cnn": {"separated_mean": 0, "global_mean": 0, "harmonic_mean": 0},
                        "lstm": {"separated_mean": 0, "global_mean": 0, "harmonic_mean": 0},
                        "transformer": {"separated_mean": 0, "global_mean": 0, "harmonic_mean": 0}}

    for sample in range(0, len(y_labels)):
        dict_data = {"cnn":{}, "lstm":{} ,"transformer":{}}
        for model in models:

            separated_mean, global_mean, harmonic_mean = calculate_means(times, y_labels[sample], predictions[model][sample])
            # print(f"\nsample {sample}:")
            # print(f"model {model}:")
            # print(f"separated_mean: {separated_mean}")
            # print(f"global_mean: {global_mean}")
            # print(f"harmonic_mean: {harmonic_mean}")

            # Calculate
            total_data_count[model]["separated_mean"] += separated_mean
            total_data_count[model]["global_mean"] += global_mean
            total_data_count[model]["harmonic_mean"] += harmonic_mean

            # Store
            dict_data[model]["separated_mean"] = separated_mean
            dict_data[model]["global_mean"] = global_mean
            dict_data[model]["harmonic_mean"] = harmonic_mean

        mean_data.append(dict_data)

    # CALCULATING THE TOTAL -----------------
    for model in models:
        total_data_count[model]["separated_mean"] = total_data_count[model]["separated_mean"]/len(y_labels)
        total_data_count[model]["global_mean"] = total_data_count[model]["global_mean"]/len(y_labels)
        total_data_count[model]["harmonic_mean"] = total_data_count[model]["harmonic_mean"]/len(y_labels)

    print("\ntotal_data_count")
    print(total_data_count)

    # # SAVING the metrics file
    # with open("global_metrics_dataset3_all_models.json", "w") as write_file:
    #     json.dump(mean_data, write_file, cls=NumpyArrayEncoder)