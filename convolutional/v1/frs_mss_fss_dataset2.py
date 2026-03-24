#!/usr/bin/env python3

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import load
import numpy as np
import pandas as pd
import scipy.stats as stats
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

    version = "v1_1"
    model_name = "cnn_" + version

    pred_cnn__data = load(version + "/data2_pred_cnn_" + version + ".npy")
    all_true = np.load("../../haptic_data/data2/y_test_data.npy")

    print(len(all_true))

    times = np.array([i for i in range(0, time_steps)])
    cnn__miss_classification_pos_transition = []
    cnn__previous_prediction_is_2nd_primitive = []
    reactivities = []
    filtered_stabilities = []
    mean_stabilities = []
    for n in range(0, int(len(all_true))):

        old_true = all_true[n]
        true = np.zeros(100)
        for i in range(0, 50):
            true[i] = old_true[0]
        for i in range(50, 100):
            true[i] = old_true[1]

        pred_cnn_ = pred_cnn__data[n]
        pull_cnn_, push_cnn_, shake_cnn_, twist_cnn_ = value_for_array(pred_cnn_, time_steps - sliding_window + 1)
        count_react = 0
        count_area = 0
        cnn__correctly_predicted = False
        changed_true = False
        previous_prediction_is_2nd_primitive = False

        for i in times:
            if i > 18:
                confidences_i = [pull_cnn_[i - 19], push_cnn_[i - 19], shake_cnn_[i - 19], twist_cnn_[i - 19]]
                pred_cnn__i = np.array(confidences_i).argmax()

                # REACTIVITY-------------------------------------------------
                if true[i] != true[i - 1]:
                    changed_true = True
                    if old_prediction == true[i]:
                        previous_prediction_is_2nd_primitive = True
                if changed_true:
                    if pred_cnn__i == true[i]:
                        changed_true = False
                        cnn__correctly_predicted = True
                    else:
                        count_react += 1
                old_prediction = pred_cnn__i

                # AREA------------------------------------
                if i >= 50:
                    count_area += confidences_i[int(true[i])]

        if cnn__correctly_predicted and not previous_prediction_is_2nd_primitive:
            reactivities.append(count_react)
            mean_stabilities.append(count_area/50)
            filtered_stabilities.append(count_area/50)
        elif not cnn__correctly_predicted:
            cnn__miss_classification_pos_transition.append(n)
            mean_stabilities.append(count_area/50)
        elif previous_prediction_is_2nd_primitive:
            cnn__previous_prediction_is_2nd_primitive.append(n)
            mean_stabilities.append(count_area/50)

    for values, title in zip ([reactivities, mean_stabilities, filtered_stabilities], ["FRS", "MSS", "FSS"]):
        mean = np.mean(values)
        std_dev = np.std(values, ddof=1)  # Sample standard deviation
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=std_dev / np.sqrt(len(values)))
        ci = [round(ci[0], 4), round(ci[1], 4)]

        print("\nTransformer " + str(version) + " " + title + " Metric:")
        print(f"mean: {round(mean, 4)}, std_dev: {round(std_dev, 4)}, 95_ci: {ci}")