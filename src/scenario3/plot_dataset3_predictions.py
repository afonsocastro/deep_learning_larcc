#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from config.definitions import ROOT_DIR
from utils import value_for_array, plot_true_shadow

if __name__ == '__main__':
    time_steps = 1500
    sliding_window = 20
    models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models = ["cnn", "lstm", "transformer"]
    models_folder = ["convolutional", "recurrent", "transformers"]
    predictions = {}
    for model, version, folder in zip(models, models_versions, models_folder):
        predictions[model] = load(ROOT_DIR + "/"+folder+"/dataset3_results/data3_pred_" + model + version + ".npy")

    data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    y_labels = data[:, :, -1]

    plot_times = np.array([i for i in range(19, time_steps)])
    real_times = np.array([i for i in range(0, time_steps)])

    for sample in range(0, len(y_labels)):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 4])  # First subplot thinner
        axes = [fig.add_subplot(gs[0])]  # first axis (thin one)
        for i in range(1, 4):
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

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
        axes[0].set_title('Dataset 3: sample ' + str(sample+1) + ' / 9')

        colors = {"pull": "blue", "push": "red", "shake": "green", "twist": "orange"}
        for ax, model in zip(axes[1:], models):
            for movement, movement_df in graph_data[model].items():
                ax.plot(movement_df["timestep"].values, movement_df[movement].values, color=colors[movement], linewidth=2, label=movement)

            if model == "cnn":
                ax.set_ylabel("CONVOLUTIONAL", fontsize=12, fontweight="bold")
                ax.legend()
            elif model == "lstm":
                ax.set_ylabel("LSTM", fontsize=12, fontweight="bold")
            elif model == "transformer":
                ax.set_ylabel("TRANSFORMER", fontsize=12, fontweight="bold")

        plt.xlabel("Timesteps")
        plt.tight_layout()
        # plt.show()
        plt.savefig("dataset3_predictions/sample_"+str(sample)+".png", bbox_inches='tight')
