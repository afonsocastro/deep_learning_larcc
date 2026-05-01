#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from config.definitions import ROOT_DIR
from utils import value_for_array, plot_shadow, grouping_segments

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

    total_ghost_transitions = 0
    total_extra_bad_transitions = 0
    total_expected_transitions = 0
    total_corrected_transitions = 0
    total_delay = 0
    total_duration = 0
    accuracy = 0

    for sample in range(0, len(y_labels)):
        print("\nsample: " +str(sample)+"/9")
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 4, 4, 4, 4])  # First subplot thinner
        axes = [fig.add_subplot(gs[0])]  # first axis (thin one)
        for i in range(1, 6):
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))
        plt.subplots_adjust(hspace=0)
        graph_data_entropy ={}
        true_labels = y_labels[sample]
        for model in models:
            pull, push, shake, twist = value_for_array(predictions[model][sample], time_steps - sliding_window + 1)
            probs = np.stack([pull, push, shake, twist], axis=1)
            arr_safe = np.where(probs < 0.001, 0.001, probs) # probabilidades abaixo de 0.001 passam a ter o valor de 0.001, para nao rebentar com logaritmos
            entropy = - np.sum(arr_safe * np.log2(arr_safe), axis=1)
            sorted_preds = np.sort(probs, axis=1)
            gap = sorted_preds[:, -1] - sorted_preds[:, -2]

            if model == "cnn":
                entropy_cnn = entropy
                gap_cnn = gap
            elif model == "lstm":
                entropy_lstm = entropy
                gap_lstm = gap
            elif model == "transformer":
                entropy_transformer = entropy
                gap_transformer = gap

        entropy_mean = (entropy_cnn + entropy_lstm + entropy_transformer) / 3
        gap_mean = (gap_cnn + gap_lstm + gap_transformer) / 3
        entropies = [entropy_cnn, entropy_lstm, entropy_transformer]
        gaps = [gap_cnn, gap_lstm, gap_transformer]

        first_derivative_gap = np.diff(gap_mean)
        second_derivative_gap = np.diff(first_derivative_gap)

        expected_transitions = plot_shadow(real_times, true_labels, axes[0])
        axes[0].set_ylabel("Ground\nTruth")
        axes[0].set_title('Dataset 3\n Sequence ' + str(sample) + ' / ' + str(len(y_labels)-1))

        axes[1].set_ylabel("Predicted\nSequence")

        axes[2].plot(plot_times[2:], second_derivative_gap, color="blue", linewidth=2, label="GAP 2nd derivative")
        axes[2].plot(plot_times[1:], first_derivative_gap, color="red", linewidth=2, label="GAP derivative")
        axes[2].set_ylabel("GAP derivative", fontsize=12, fontweight="bold")
        axes[2].legend()

        axes[3].plot(plot_times, gap_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        axes[3].plot(plot_times, gap_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        axes[3].plot(plot_times, gap_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        axes[3].plot(plot_times, gap_mean, color="blue", linewidth=2, label="Mean")
        axes[3].set_ylabel("GAP", fontsize=12, fontweight="bold")
        axes[3].legend()

        axes[4].plot(plot_times, entropy_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        axes[4].plot(plot_times, entropy_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        axes[4].plot(plot_times, entropy_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        axes[4].plot(plot_times, entropy_mean, color="blue", linewidth=2, label="Mean")
        axes[4].set_ylabel("ENTROPY", fontsize=12, fontweight="bold")
        axes[4].legend()

        # --- obter valores ---
        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(predictions["cnn"][sample],
                                                                   time_steps - sliding_window + 1)
        pull_trans, push_trans, shake_trans, twist_trans = value_for_array(predictions["transformer"][sample],
                                                                           time_steps - sliding_window + 1)

        # --- somar ---
        pull_sum = pull_cnn + pull_trans
        push_sum = push_cnn + push_trans
        shake_sum = shake_cnn + shake_trans
        twist_sum = twist_cnn + twist_trans

        # # --- máscara ---
        mask = shake_sum > 0.6

        pull_sum[mask] = 0.04
        push_sum[mask] = 0.04
        twist_sum[mask] = 0.04
        shake_sum[mask] = 1.88

        # --- reconstruir probs ---
        probs = np.stack([pull_sum, push_sum, shake_sum, twist_sum], axis=1)

        # --- métricas ---
        arr_safe = np.where(probs < 0.001, 0.001, probs)
        entropy = - np.sum(arr_safe * np.log2(arr_safe), axis=1)

        sorted_preds = np.sort(probs, axis=1)
        gap = sorted_preds[:, -1] - sorted_preds[:, -2]

        axes[5].plot(plot_times, gap, color="orange", linewidth=2, label="GAP")
        axes[5].plot(plot_times, entropy, color="blue", linewidth=2, label="Entropy")
        axes[5].set_ylabel("CNN + Transformer", fontsize=12, fontweight="bold")
        axes[5].legend()
        plt.xlabel("Timesteps")
        plt.tight_layout()
        plt.show()
        # plt.savefig("dataset3_pred/sample_"+str(sample)+".png", bbox_inches='tight')
