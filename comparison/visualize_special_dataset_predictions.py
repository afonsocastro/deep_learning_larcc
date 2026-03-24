#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist

def plot_true_shadow(ts, t, a):
    last = len(ts)
    start = 0
    for i in ts:
        if (i != 0 and t[i] != t[i-1]) or i == last-1:
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
    time_steps = 350
    sliding_window = 20
    models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models = ["cnn", "lstm", "transformer"]
    predictions = {}
    for model, version in zip(models, models_versions):
        predictions[model] = load("dataset3_pred/data3_pred_" + model + version + ".npy")
        # predictions[model] = load("dataset3_old_results/data3_pred_" + model + version + ".npy")

    y_data = np.load("../haptic_data/data3/y_test_data.npy")
    y_labels = np.repeat(y_data, 50, axis=1)
    print("\ny_labels.shape")
    print(y_labels.shape)

    # data = np.load("../haptic_data/data3_old/global_normalized_data.npy")
    # y_labels = data[:, :, -1]

    plot_times = np.array([i for i in range(19, time_steps)])
    real_times = np.array([i for i in range(0, time_steps)])

    for sample in range(0, len(y_labels)):
        # Create figure with GridSpec for custom subplot heights
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 4])  # First subplot thinner
        # axes = [fig.add_subplot(gs[i]) for i in range(4)]
        # Create axes and share x-axis
        axes = [fig.add_subplot(gs[0])]  # first axis (thin one)
        for i in range(1, 4):
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

        # fig, axes = plt.subplots(len(models), 1, figsize=(16, 10), sharex=True, squeeze=False)
        plt.subplots_adjust(hspace=0)
        graph_data ={}
        true_labels = y_labels[sample]
        for model in models:
            pull, push, shake, twist = value_for_array(predictions[model][sample], time_steps - sliding_window + 1)
            graph_data[model] = {"pull": pd.DataFrame({'timestep': plot_times, 'pull': pull}),
                                 "push": pd.DataFrame({'timestep': plot_times, 'push': push}),
                                 "shake": pd.DataFrame({'timestep': plot_times, 'shake': shake}),
                                 "twist": pd.DataFrame({'timestep': plot_times, 'twist': twist})}

        plot_true_shadow(real_times, true_labels, axes[0])
        axes[0].set_ylabel("Ground\nTruth")
        axes[0].set_title('Dataset 3: sample ' + str(sample+1) + ' / 113')

        for ax, model in zip(axes[1:], models):
            df = graph_data[model]  # Get data for the current model
            # plot_true_shadow(real_times, true_labels, ax)
            for movement, color in zip(["pull", "push", "shake", "twist"], ["blue", "red", "green", "orange"]):
                ax.plot(df[movement]["timestep"], df[movement][movement], color = color, linewidth=2, label=movement)

            # Regra para inicio de transicao
            # if model == "cnn":
            #     for i in range(19, 481):
            #         if max(df["pull"]["pull"].loc[i], df["push"]["push"].loc[i], df["shake"]["shake"].loc[i], df["twist"]["twist"].loc[i]) <0.9:
            #             ax.axvspan(i+19-1, i+19, color="grey", alpha=0.4, lw=0)

            # Add the vertical line to the plot
            for i in range(1,7):
                ax.axvline(x=50 * i, linestyle="--")

            if model == "cnn":
                ax.set_ylabel("CONVOLUTIONAL", fontsize=12, fontweight="bold")
                ax.legend()
            elif model == "lstm":
                ax.set_ylabel("LSTM", fontsize=12, fontweight="bold")
            elif model == "transformer":
                ax.set_ylabel("TRANSFORMER", fontsize=12, fontweight="bold")

        plt.xlabel("Timesteps")
        plt.tight_layout()
        plt.show()
        # plt.savefig("dataset3_pred/sample_"+str(sample)+".png", bbox_inches='tight')
