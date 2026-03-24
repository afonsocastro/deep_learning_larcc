#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from prettytable import PrettyTable
import statistics



def compute_segment_metrics(gt, pred):
    """
    Compute segment-level Accuracy, Precision, Recall, F1 using IoU approach.

    Parameters:
    - gt: list of tuples (label, start, end) -- ground truth (no gray)
    - pred: list of tuples (label, start, end) -- predicted (label 4 = gray)

    Returns:
    - accuracy, precision, recall, f1
    """
    tp_len = 0  # True positive length
    total_pred_len = 0
    total_gt_len = 0

    for g_label, g_start, g_end in gt:
        total_gt_len += g_end - g_start + 1

        # Find predicted segments that overlap with this gt segment
        overlap_tp = 0
        overlap_pred_len = 0

        for p_label, p_start, p_end in pred:
            # Skip gray areas
            if p_label == 4:
                continue

            # Compute overlap
            start = max(g_start, p_start)
            end = min(g_end, p_end)
            if start <= end:
                overlap_len = end - start + 1
                overlap_pred_len += p_end - p_start + 1  # sum predicted length
                if p_label == g_label:
                    overlap_tp += overlap_len

        tp_len += overlap_tp
        total_pred_len += overlap_pred_len

    # Handle cases where there is no prediction
    if total_pred_len == 0:
        precision = 0.0
    else:
        precision = tp_len / total_pred_len

    recall = tp_len / total_gt_len
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = tp_len / total_gt_len
    return accuracy, precision, recall, f1



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
            (t, d) for (t, d) in real_trans if start - 200 < t < start + 300
        ]

        if matching_real_transitions:
            # First one is correct
            correct_real_transitions.append(matching_real_transitions[0])
            real_trans.remove(matching_real_transitions[0])
            # The rest are extra
            if len(matching_real_transitions) > 1:
                for i in range(1, len(matching_real_transitions)):
                    # tuplex = (matching_real_transitions[i][0], matching_real_transitions[i][1])
                    extra_bad_transitions.append(matching_real_transitions[i])
                    real_trans.remove(matching_real_transitions[i])
        else:
            # No real transition in this window
            ghost_transitions += 1

    if real_transitions:
        extra_bad_transitions.extend(real_transitions[:])

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


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist

def plot_true_shadow(ts, t, a):
    last = len(ts)
    start = 0
    expected_transitions = [19]
    for i in ts:
        if (i != 0 and t[i] != t[i-1]) or i == last-1:
            expected_transitions.append(int(i))
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
    return expected_transitions[:-1]

def grouping_segments(categories_list, delay):
    predicted_segments = []
    start_idx = 0
    current_label = categories_list[0]
    for i in range(1, len(categories_list)):
        if categories_list[i] != current_label:
            predicted_segments.append((int(current_label), start_idx + delay, i - 1 + delay))
            start_idx = i
            current_label = categories_list[i]
    predicted_segments.append((int(current_label), start_idx + delay, len(categories_list) - 1 + delay))
    return predicted_segments

# def plot_predicted_transition_shadow(ts, second_derivative_gap, predictions, sample, a):
#     last = len(ts) #350
#     first_that_count = last - len(second_derivative_gap) #21
#     start = first_that_count
#     counting = False
#     count = 0
#     previous_color = None
#     predicted_sequence = []
#     switch_color_case = { "blue": 0, "red": 1, "green": 2, "orange": 3, "grey": 4}
#     for i in ts[first_that_count+10:]:
#         std_dev = np.std(second_derivative_gap[i-first_that_count-10:i-first_that_count])
#         if std_dev > 0.05:
#             end = i
#             color = "grey"
#             alpha = 0.9
#         else:
#             end = i
#             pull_pred = 0
#             push_pred = 0
#             shake_pred = 0
#             twist_pred = 0
#             for model in ["cnn", "lstm", "transformer"]:
#                 pull_pred += predictions[model][sample][i-first_that_count][0]
#                 push_pred += predictions[model][sample][i-first_that_count][1]
#                 shake_pred += predictions[model][sample][i-first_that_count][2]
#                 twist_pred += predictions[model][sample][i-first_that_count][3]
#             sum_preds = [pull_pred, push_pred, shake_pred, twist_pred]
#
#             pred_primitive = sum_preds.index(max(sum_preds))
#             if pred_primitive == 0:
#                 color = "blue"
#             elif pred_primitive == 1:
#                 color = "red"
#             elif pred_primitive == 2:
#                 color = "green"
#             elif pred_primitive == 3:
#                 color = "orange"
#             alpha = 0.4
#
#         if color != previous_color or count == 10:
#             if counting:
#                 if count == 10:
#                     count = 0
#                     counting = False
#                 else:
#                     color = oldest_color
#                     count = 0
#                     counting = True
#                     a.axvspan(start, end, color="grey", alpha=0.9, lw=0)
#                     for i in range(start, end):
#                         predicted_sequence.append(switch_color_case["grey"])
#                     start = end
#                     previous_color = "grey"
#             else:
#                 counting = True
#                 oldest_color = previous_color
#                 previous_color = color
#
#         if counting:
#             count += 1
#         else:
#             a.axvspan(start, end, color=color, alpha=alpha, lw=0)
#             for i in range(start, end):
#                 predicted_sequence.append(switch_color_case[color])
#             start = end
#             previous_color = color
#     predicted_sequence.append(switch_color_case[color])
#     predicted_segments = grouping_segments(predicted_sequence, delay=first_that_count)
#     return predicted_segments


def plot_predicted_transition_shadow(ts, entropy_combined_mean, predictions, sample, a):
    last = len(ts) #6000
    first_that_count = last - len(entropy_combined_mean) #19
    start = first_that_count
    counting = False
    count = 0
    previous_color = None
    oldest_color = None
    predicted_sequence = []
    switch_color_case = { "blue": 0, "red": 1, "green": 2, "orange": 3, "grey": 4}
    for i in ts[first_that_count:]:
        if entropy_combined_mean[i-first_that_count] > 1:
            end = i
            color = "grey"
            alpha = 0.9
        else:
            end = i
            pull_pred = 0
            push_pred = 0
            shake_pred = 0
            twist_pred = 0
            weight = 1
            for model in ["cnn", "lstm", "transformer"]:
                # if model == "cnn":
                #     weight = 0.6
                # elif model == "transformer":
                #     weight = 0.3
                # elif model == "lstm":
                #     weight = 0.1

                pull_pred += weight * predictions[model][sample][i-first_that_count][0]
                push_pred += weight * predictions[model][sample][i-first_that_count][1]
                shake_pred += weight * predictions[model][sample][i-first_that_count][2]
                twist_pred += weight * predictions[model][sample][i-first_that_count][3]


            # pull_pred += predictions["transformer"][sample][i-first_that_count][0]
            # push_pred += predictions["transformer"][sample][i-first_that_count][1]
            # shake_pred += predictions["transformer"][sample][i-first_that_count][2]
            # twist_pred += predictions["transformer"][sample][i-first_that_count][3]

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

        if (color != previous_color or count == 20) and (previous_color is not None):
            counting = True
            if count == 20 and color == previous_color:
                a.axvspan(start, end, color=color, alpha=alpha, lw=0)
                for i in range(start, end):
                    predicted_sequence.append(switch_color_case[color])
                start = end
                counting = False
                count = 0
                oldest_color = color

            elif color != previous_color and color != oldest_color and oldest_color != previous_color and 0 < count:
                a.axvspan(start, end, color="grey", alpha=0.9, lw=0)
                for i in range(start, end):
                    predicted_sequence.append(switch_color_case["grey"])
                start = end
                oldest_color = "grey"
                count = 0

            elif color != previous_color and color == oldest_color and 0 < count :
                a.axvspan(start, end, color=color, alpha=alpha, lw=0)
                for i in range(start, end):
                    predicted_sequence.append(switch_color_case[color])
                start = end
                oldest_color = color
                count = 0
            else:
                oldest_color = previous_color
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


def conditional_mean(entropy_cnn, entropy_lstm, entropy_transformer):
    result = []
    for cnn, lstm, trans in zip(entropy_cnn, entropy_lstm, entropy_transformer):
        values = [cnn, lstm, trans]
        count_above_1 = sum(v > 1 for v in values)

        if count_above_1 == 1:
            # Ignore the one that's > 1
            filtered = [v for v in values if v <= 1]
            result.append(sum(filtered) / len(filtered))
        else:
            # Use all three values
            result.append(sum(values) / 3)
    return result


if __name__ == '__main__':

    time_steps = 6000
    sliding_window = 20
    entropy_epsilon = 0.05
    models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models = ["cnn", "lstm", "transformer"]
    predictions = {}
    t = PrettyTable(['Sample', 'Acc', 'Prec', 'Rec', 'F1', 'Ghost' , 'extra bad', 'Corr', 'Exp', 'Delay', 'Duration'])
    final_table_list = []
    remove_idx = [2, 7, 11, 14, 15, 17]
    for model, version in zip(models, models_versions):
        predictions[model] = load("dataset3_old_results/data3_pred_" + model + version + ".npy")
        predictions[model] = np.delete(predictions[model], remove_idx, axis=0)

    y_data = np.load("../haptic_data/data3_old/global_normalized_data.npy")
    y_data = np.delete(y_data, remove_idx, axis=0)
    y_labels = y_data[:, :, -1]

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
        print("sample: " +str(sample)+"/"+str(len(y_labels)-1))
        # Create figure with GridSpec for custom subplot heights
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 4, 4, 4])  # First subplot thinner
        # axes = [fig.add_subplot(gs[i]) for i in range(4)]
        # Create axes and share x-axis
        axes = [fig.add_subplot(gs[0])]  # first axis (thin one)
        for i in range(1, 5):
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

        # fig, axes = plt.subplots(len(models), 1, figsize=(16, 10), sharex=True, squeeze=False)
        plt.subplots_adjust(hspace=0)
        graph_data_entropy ={}
        true_labels = y_labels[sample]
        for model in models:
            pull, push, shake, twist = value_for_array(predictions[model][sample], time_steps - sliding_window + 1)
            probs = np.stack([pull, push, shake, twist], axis=1)  # shape: (481, 4)

            arr_safe = np.where(probs < entropy_epsilon, entropy_epsilon, probs)
            entropy = - np.sum(arr_safe * np.log2(arr_safe), axis=1)  # shape: (481,)

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


        entropy_w_sum = conditional_mean(entropy_cnn, entropy_lstm, entropy_transformer)
        entropy_mean = (entropy_cnn + entropy_lstm + entropy_transformer) / 3
        # gap_mean = (gap_cnn + gap_transformer) / 2

        # first_derivative_gap = np.diff(gap_mean)
        # second_derivative_gap = np.diff(first_derivative_gap)
        # third_derivative_gap = np.diff(second_derivative_gap)


        # predicted_sequence = plot_predicted_transition_shadow(real_times, entropy_cnn, predictions, sample, axes[1]) #CNN solo
        # predicted_sequence = plot_predicted_transition_shadow(real_times, entropy_lstm, predictions, sample, axes[1]) #LSTM solo
        # predicted_sequence = plot_predicted_transition_shadow(real_times, entropy_transformer, predictions, sample, axes[1]) #Transformer solo
        predicted_sequence = plot_predicted_transition_shadow(real_times, entropy_mean, predictions, sample, axes[1]) #3 models combined solo
        # predicted_sequence = plot_predicted_transition_shadow(real_times, entropy_w_sum, predictions, sample, axes[1]) #proposed ensemble

        ground_truth_sequence = grouping_segments(true_labels, delay=0)
        # print("predicted_sequence")
        # print(predicted_sequence)
        real_transitions = [(start, end-start) for label, start, end in predicted_sequence if label == 4]
        # print("ground_truth_sequence")
        # print(ground_truth_sequence)

        metrics = compute_segment_metrics(ground_truth_sequence, predicted_sequence )
        # print("\nmetrics")
        # print(metrics)

        expected_transitions = plot_true_shadow(real_times, true_labels, axes[0])
        # print("\nexpected_transitions")
        # print(expected_transitions)
        # print("real transitions")
        # print(real_transitions)


        axes[0].set_ylabel("Ground\nTruth")
        axes[0].set_title('Dataset 3: sample ' + str(sample+1) + ' / 17')
        exp, corr, ghost, extra_bad, sdel, sdur = computing_transient_metrics(expected_transitions, real_transitions)

        total_expected_transitions += exp
        total_corrected_transitions += corr
        total_ghost_transitions += ghost
        total_extra_bad_transitions += extra_bad
        total_delay += sdel
        total_duration += sdur

        # axes[1].set_ylabel("Predicted\nSequence")
        #
        # axes[2].plot(plot_times[2:], second_derivative_gap, color="blue", linewidth=2, label="GAP derivative")
        # # axes[2].plot(plot_times[3:], third_derivative_gap, color="blue", linewidth=2, label="GAP derivative")
        # # for i in range(1, n_samples):
        # #     axes[2].axvline(x=50 * i, linestyle="--")
        # axes[2].set_ylabel("GAP derivative", fontsize=12, fontweight="bold")
        # axes[2].legend()
        #
        # axes[3].plot(plot_times, gap_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        # axes[3].plot(plot_times, gap_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        # axes[3].plot(plot_times, gap_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        # # axes[3].plot(plot_times, gap_mean, color="blue", linewidth=2, label="Mean")
        #
        # axes[3].set_ylabel("GAP", fontsize=12, fontweight="bold")
        # axes[3].legend()
        #
        # axes[4].plot(plot_times, entropy_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        # axes[4].plot(plot_times, entropy_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        # axes[4].plot(plot_times, entropy_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        # # axes[4].plot(plot_times, entropy_mean, color="blue", linewidth=2, label="Mean")
        # axes[4].plot(plot_times, entropy_w_sum, color="blue", linewidth=2, label="w Sum")
        #
        #
        # axes[4].set_ylabel("ENTROPY", fontsize=12, fontweight="bold")
        # axes[4].legend()
        # plt.xlabel("Timesteps")
        # plt.tight_layout()
        # plt.show()
        # t = PrettyTable(
        #     ['Sample', 'Acc', 'Prec', 'Rec', 'F1', 'Ghost', 'extra bad', 'Corr', 'Exp', 'Delay', 'Duration'])
        final_table_list.append([round(metrics[0], 3), round(metrics[1], 3), round(metrics[2], 3), round(metrics[3], 3), ghost, extra_bad, corr, exp, round(sdel/corr, 3), round(sdur/corr, 3) ])
        t.add_row([sample, round(metrics[0], 3), round(metrics[1], 3), round(metrics[2], 3), round(metrics[3], 3), ghost, extra_bad, corr, exp, round(sdel/corr, 3), round(sdur/corr, 3) ])

    means = [round(statistics.mean(col), 3) for col in zip(*final_table_list)]
    t.add_row(["TOTAL", means[0], means[1], means[2], means[3], means[4], means[5], means[6], means[7], means[8], means[9]])
    print(t)
    ratio_ghost = total_ghost_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    ratio_extra_bad = total_extra_bad_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    mean_duration = total_duration / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_delay = total_delay / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_accuracy = accuracy / len(y_labels)
    print(f"\nRatio of ghost transitions / expected: {ratio_ghost:.3f}")
    print(f"Ratio of extra bad transitions / expected: {ratio_extra_bad:.3f}")
    print(f"Mean duration of correct real transitions: {mean_duration:.2f}")
    print(f"Mean delay between expected and real correct transitions: {mean_delay:.2f}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}")