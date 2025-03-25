#!/usr/bin/env python3

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

def plot_true_shadow(ts, t, a):
    start = 0
    for i in ts:
        if (i != 0 and t[i] != t[i-1]) or i == 5999:
            end = i
            if t[i-1] == 0:
                color = "blue"
            elif t[i-1] == 1:
                color = "red"
            elif t[i-1] == 2:
                color = "green"
            elif t[i-1] == 3:
                color = "orange"
            a.axvspan(start, end, color=color, alpha=0.2, lw=0)
            start = end


if __name__ == '__main__':
    time_steps = 6000
    sliding_window = 20
    models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models = ["cnn", "lstm", "transformer"]
    predictions = {}
    for model, version in zip(models, models_versions):
        predictions[model] = load("dataset3_results/data3_pred_" + model + version + ".npy")

    data = np.load("../haptic_data/data3/global_normalized_data.npy")
    y_labels = data[:, :, -1]

    plot_times = np.array([i for i in range(19, time_steps)])
    real_times = np.array([i for i in range(0, time_steps)])

    for sample in range(0, len(y_labels)):
        fig, axes = plt.subplots(len(models), 1, figsize=(16, 10), sharex=True, squeeze=False)
        plt.subplots_adjust(hspace=0)
        axes = axes.flatten()
        graph_data ={}
        true_labels = y_labels[sample]
        for model in models:
            pull, push, shake, twist = value_for_array(predictions[model][sample], time_steps - sliding_window + 1)
            graph_data[model] = {"pull": pd.DataFrame({'timestep': plot_times, 'pull': pull}),
                                 "push": pd.DataFrame({'timestep': plot_times, 'push': push}),
                                 "shake": pd.DataFrame({'timestep': plot_times, 'shake': shake}),
                                 "twist": pd.DataFrame({'timestep': plot_times, 'twist': twist})}

        for ax, model in zip(axes, models):
            df = graph_data[model]  # Get data for the current model
            plot_true_shadow(real_times, true_labels, ax)
            for movement, color in zip(["pull", "push", "shake", "twist"], ["blue", "red", "green", "orange"]):
                ax.plot(df[movement]["timestep"], df[movement][movement], color = color, linewidth=2, label=movement)
            if model == "cnn":
                ax.set_ylabel("CONVOLUTIONAL")
                ax.legend()
                ax.set_title('Dataset 3: sample ' + str(sample) + ' / 17')
            elif model == "lstm":
                ax.set_ylabel("LSTM")
            elif model == "transformer":
                ax.set_ylabel("TRANSFORMER")

        plt.xlabel("Timesteps")
        plt.tight_layout()
        # plt.show()
        plt.savefig("dataset3_results/sample_"+str(sample)+".png", bbox_inches='tight')
