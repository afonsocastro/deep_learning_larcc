#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from config.definitions import ROOT_DIR
from utils import value_for_array, plot_shadow, grouping_segments

def computing_transient_metrics (expected_trans, real_trans):
    correct_real_transitions = []
    ghost_transitions = 0
    extra_bad_transitions = []

    # Step through each expected transition
    for i, expected_time in enumerate(expected_trans):
        # Define the window for this expected transition
        start = expected_time
        end = expected_trans[i + 1] if i + 1 < len(expected_trans) else float('inf')

        # Find real transitions that fall into this window
        matching_real_transitions = [
            (t, d) for (t, d) in real_trans if start < t < end
        ]

        if matching_real_transitions:
            # First one is correct
            correct_real_transitions.append(matching_real_transitions[0])
            # The rest are extra
            extra_bad_transitions.extend(matching_real_transitions[1:])
        else:
            # No real transition in this window
            ghost_transitions += 1

    # print("correct_real_transitions")
    # print(correct_real_transitions)
    # print("extra_bad_transitions")
    # print(extra_bad_transitions)
    # print("ghost_transitions")
    # print(ghost_transitions)

    # Now compute the metrics
    total_expected = len(expected_trans)
    total_correct = len(correct_real_transitions)
    total_extra_bad = len(extra_bad_transitions)

    # 2. Ratio of ghost / total_expected
    ratio_ghost = ghost_transitions / total_expected if total_expected > 0 else float('inf')

    # 3. Ratio of extra bad / correct
    ratio_extra_bad = total_extra_bad / total_expected if total_expected > 0 else float('inf')

    # 4. Mean duration of correct real transitions
    sum_duration = sum(d for _, d in correct_real_transitions)
                     # / total_correct) if total_correct > 0 else 0

    # 5. Mean delay between expected and real transitions
    sum_delay = sum(rt - expected_trans[i] for i, (rt, _) in enumerate(correct_real_transitions))
                     # ) / total_correct if total_correct > 0 else 0

    # Output
    # print(f"Ratio of ghost transitions / expected: {ratio_ghost:.3f}")
    # print(f"Ratio of extra bad transitions / expected: {ratio_extra_bad:.3f}")
    # print(f"Mean duration of correct real transitions: {mean_duration:.2f}")
    # print(f"Mean delay between expected and real correct transitions: {mean_delay:.2f}")
    return total_expected, total_correct, ghost_transitions, total_extra_bad, sum_delay, sum_duration


def plot_predicted_transition_shadow(ts, models, second_derivative_gap, predictions, sample, a):
    last = len(ts) #350
    first_that_count = last - len(second_derivative_gap) #21
    start = first_that_count
    counting = False
    count = 0
    previous_color = None
    predicted_sequence = []
    switch_color_case = { "blue": 0, "red": 1, "green": 2, "orange": 3, "grey": 4}
    for i in ts[first_that_count+10:]:
        std_dev = np.std(second_derivative_gap[i-first_that_count-10:i-first_that_count])
        # if std_dev > 0.05:
        if std_dev > 0.3:
            end = i
            color = "grey"
            alpha = 0.9
        else:
            end = i
            pull_pred = 0
            push_pred = 0
            shake_pred = 0
            twist_pred = 0
            for model in models:
                pull_pred += predictions[model][sample][i-first_that_count][0]
                push_pred += predictions[model][sample][i-first_that_count][1]
                shake_pred += predictions[model][sample][i-first_that_count][2]
                twist_pred += predictions[model][sample][i-first_that_count][3]
            sum_preds = [pull_pred, push_pred, shake_pred, twist_pred]

            pred_primitive = sum_preds.index(max(sum_preds))
            if pred_primitive == 0:
                color = "blue"
            elif pred_primitive == 1:
                color = "red"
            elif pred_primitive == 2:
                color = "green"
            elif pred_primitive == 3:
                color = "orange"
            alpha = 0.4

        if color != previous_color or count == 5:
            if counting:
                if count == 5:
                    count = 0
                    counting = False
                else:
                    color = oldest_color
                    count = 0
                    counting = True
                    a.axvspan(start, end, color="grey", alpha=0.9, lw=0)
                    for i in range(start, end):
                        predicted_sequence.append(switch_color_case["grey"])
                    start = end
                    previous_color = "grey"
            else:
                counting = True
                oldest_color = previous_color
                previous_color = color

        if counting:
            count += 1
        else:
            a.axvspan(start, end, color=color, alpha=alpha, lw=0)
            for i in range(start, end):
                predicted_sequence.append(switch_color_case[color])
            start = end
            previous_color = color
    predicted_sequence.append(switch_color_case[color])
    predicted_segments = grouping_segments(predicted_sequence, delay=first_that_count)
    return predicted_segments


def compute_predicted_transitions(y_labels, sample, predictions, ):
    print("\nsample: " + str(sample) + "/9")

    true_labels = y_labels[sample]
    ground_truth_sequence = grouping_segments(true_labels, delay=0)
    print("ground_truth_sequence")
    print(ground_truth_sequence)

    pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(predictions["cnn"][sample],
                                                               time_steps - sliding_window + 1)
    pull_trans, push_trans, shake_trans, twist_trans = value_for_array(predictions["transformer"][sample],
                                                                       time_steps - sliding_window + 1)
    pull_sum = pull_cnn + pull_trans
    push_sum = push_cnn + push_trans
    shake_sum = shake_cnn + shake_trans
    twist_sum = twist_cnn + twist_trans

    mask = shake_sum > 0.6
    pull_sum[mask] = 0.04
    push_sum[mask] = 0.04
    twist_sum[mask] = 0.04
    shake_sum[mask] = 1.88
    probs = np.stack([pull_sum, push_sum, shake_sum, twist_sum], axis=1)
    arr_safe = np.where(probs < 0.001, 0.001, probs)
    entropy = - np.sum(arr_safe * np.log2(arr_safe), axis=1)

    first_that_count = 19
    start = first_that_count
    counting = False
    previous_primitive = None
    count = 0
    predicted_sequence = []
    for i in real_times[first_that_count:]:
        if entropy[i - first_that_count] > 0.5:
            end = i
            pred_primitive = 4
            count = True
        else:
            end = i
            prob = probs[i - first_that_count]
            pred_primitive = np.argmax(prob)
            count = False

        # if pred_primitive == previous_primitive:
        #     counting = True

        if counting:
            count += 1
            if count == 10:
                for i in range(start, end):
                    predicted_sequence.append(4)
                start = end
                counting = False
                count = 0
        else:
            for i in range(start, end):
                predicted_sequence.append(pred_primitive)
            start = end
            counting = 0
        previous_primitive = pred_primitive

    predicted_sequence.append(pred_primitive)
    predicted_segments = grouping_segments(predicted_sequence, delay=first_that_count)
    return  predicted_segments


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

        # SOLO CNN
        # predicted_sequence = plot_predicted_transition_shadow(real_times, models, second_derivative_gap, predictions, sample, axes[1])
        # ground_truth_sequence = grouping_segments(true_labels, delay=0)
        # print("predicted_sequence")
        # print(predicted_sequence)
        # print("ground_truth_sequence")
        # print(ground_truth_sequence)
        # # real_transitions, sample_accuracy = cnn_solo_results(real_times, entropies, gaps, predictions, sample, true_labels, axes[1])
        # real_transitions = [(start, end-start) for label, start, end in predicted_sequence if label == 4]

        expected_transitions = plot_shadow(real_times, true_labels, axes[0])
        axes[0].set_ylabel("Ground\nTruth")
        axes[0].set_title('Dataset 3: sample ' + str(sample+1) + ' / 9')

        # exp, corr, ghost, extra_bad, sdel, sdur = computing_transient_metrics(expected_transitions, real_transitions)
        # total_expected_transitions += exp
        # total_corrected_transitions += corr
        # total_ghost_transitions += ghost
        # total_extra_bad_transitions += extra_bad
        # total_delay += sdel
        # total_duration += sdur

        # accuracy += sample_accuracy
        # predicted_sequence = compute_predicted_transitions(y_labels, sample, predictions)
        # pred_transitions = plot_shadow(real_times, predicted_sequence, axes[1])
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


    # ratio_ghost = total_ghost_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    # ratio_extra_bad = total_extra_bad_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    # mean_duration = total_duration / total_corrected_transitions if total_corrected_transitions > 0 else 0
    # mean_delay = total_delay / total_corrected_transitions if total_corrected_transitions > 0 else 0
    # mean_accuracy = accuracy / len(y_labels)
    # print(f"Ratio of ghost transitions / expected: {ratio_ghost:.3f}")
    # print(f"Ratio of extra bad transitions / expected: {ratio_extra_bad:.3f}")
    # print(f"Mean duration of correct real transitions: {mean_duration:.2f}")
    # print(f"Mean delay between expected and real correct transitions: {mean_delay:.2f}")
    # print(f"Mean Accuracy: {mean_accuracy:.2f}")