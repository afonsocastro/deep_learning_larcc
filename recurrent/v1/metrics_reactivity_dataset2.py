#!/usr/bin/env python3
from time import sleep

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist


if __name__ == '__main__':
    time_steps = 100
    sliding_window = 20

    version = "v1_4"
    model_name = "lstm_" + version

    pred_lstm_data = load(version + "/data2_pred_lstm_" + version + ".npy")
    all_true = np.load("../../haptic_data/data2/y_test_data.npy")

    print(len(all_true))

    times = np.array([i for i in range(0, time_steps)])
    total_pull_seq2label, total_push_seq2label, total_shake_seq2label, total_twist_seq2label = 0, 0, 0, 0
    total_count_lstm = 0

    for n in range(0, int(len(all_true))):

        old_true = all_true[n]
        true = np.zeros(100)
        for i in range(0, 50):
            true[i] = old_true[0]
        for i in range(50, 100):
            true[i] = old_true[1]

        pred_lstm = pred_lstm_data[n]
        pull_lstm, push_lstm, shake_lstm, twist_lstm = value_for_array(pred_lstm, time_steps - sliding_window + 1)
        count_lstm = 0
        changed_lstm = False

        for i in times:
            if i > 18:
                pred_lstm_i = np.array([pull_lstm[i - 19], push_lstm[i - 19], shake_lstm[i - 19],
                                             twist_lstm[i - 19]]).argmax()
                if true[i] != true[i - 1]:
                    changed_lstm = True
                if changed_lstm:
                    if pred_lstm_i == true[i]:
                        changed_lstm = False
                    else:
                        count_lstm += 1

                old_lstm_pred = pred_lstm_i

        total_count_lstm += count_lstm
    lstm_metric = total_count_lstm / len(all_true)

    print("\nLSTM " + str(version) + " Reactivity Metric:")
    print(lstm_metric)
