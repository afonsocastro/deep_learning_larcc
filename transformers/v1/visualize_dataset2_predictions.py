#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def value_for_array(data, n, timesteps):
    pull = np.array([data[n][j][0] for j in range(timesteps)])
    push = np.array([data[n][j][1] for j in range(timesteps)])
    shake = np.array([data[n][j][2] for j in range(timesteps)])
    twist = np.array([data[n][j][3] for j in range(timesteps)])

    return pull, push, shake, twist


def string_result(true):
    results = []

    for i in range(0, 2):
        if true[i] == 0:
            result = "PULL"
        elif true[i] == 1:
            result = "PUSH"
        elif true[i] == 2:
            result = "SHAKE"
        elif true[i] == 3:
            result = "TWIST"
        results.append(result)

    return results


if __name__ == '__main__':
    version = "v1_4"
    model_name = "transformer_" + version

    pred_transformer = load(version + "/data2_pred_lstm_" + version + ".npy")
    y_test = np.load("../../haptic_data/data2/y_test_data.npy")

    times = np.array([i for i in range(19, 100)])

    for n in range(0, len(pred_transformer)):

        fig = plt.figure(figsize=(12, 8))

        # LSTM GRAPH
        pull_transformer, push_transformer, shake_transformer, twist_transformer = value_for_array(pred_transformer, n, 81)

        df11 = pd.DataFrame({'timestep': times, 'pull_cnn': pull_transformer})
        df22 = pd.DataFrame({'timestep': times, 'push_cnn': push_transformer})
        df33 = pd.DataFrame({'timestep': times, 'shake_cnn': shake_transformer})
        df44 = pd.DataFrame({'timestep': times, 'twist_cnn': twist_transformer})

        plt.plot(df11.timestep, df11.pull_transformer, color="blue", label='pull', linewidth=3)
        plt.plot(df22.timestep, df22.push_transformer, color='red', label='push', linewidth=3)
        plt.plot(df33.timestep, df33.shake_transformer, color='green', label='shake', linewidth=3)
        plt.plot(df44.timestep, df44.twist_transformer, color='orange', label='twist', linewidth=3)

        # Add the vertical line to the plot
        plt.axvline(x=50, linestyle="--")

        results_true = string_result(y_test[n])
        plt.title('Transformer ' + version + ' output confidences/timestep. True Expected: ' + results_true[0] + ' and ' + results_true[1])
        plt.title("Ground Truth:    " + results_true[0] + '  -->  ' + results_true[1], fontdict={"fontsize": 16, "fontweight": "bold"})
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()

