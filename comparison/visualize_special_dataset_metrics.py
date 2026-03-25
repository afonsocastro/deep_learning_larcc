#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

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


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist

def plot_true_shadow(ts, t, a):
    last = len(ts)
    start = 0
    expected_transitions = []
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


if __name__ == '__main__':
    time_steps = 1500
    # n_samples = 7
    sliding_window = 20
    entropy_epsilon = 0.01
    # models_versions = ["_v1_1", "_v1_2", "_v1_1"]
    models_versions = ["_v1_1"]
    # models = ["cnn", "lstm", "transformer"]
    models = ["cnn"]
    predictions = {}
    for model, version in zip(models, models_versions):
        predictions[model] = load("dataset3_pred/data3_pred_" + model + version + ".npy")
        # predictions[model] = load("dataset3_old_results/data3_pred_" + model + version + ".npy")

    # y_data = np.load("../haptic_data/data3/y_test_data.npy")
    # y_labels = np.repeat(y_data, 50, axis=1)
    data = np.load("../haptic_data/full_timewindow/data/normalized_data_15s.npy")
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

    for sample in range(3, len(y_labels)):
        print("sample: " +str(sample)+"/9")
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

        # entropy_mean = (entropy_cnn + entropy_lstm + entropy_transformer) / 3
        entropy_mean = entropy_cnn
        # # gap_mean = (gap_cnn + gap_lstm + gap_transformer) / 3
        # gap_mean = (gap_cnn + gap_transformer) / 2
        gap_mean = gap_cnn
        #
        # entropies = [entropy_cnn, entropy_lstm, entropy_transformer]
        # gaps = [gap_cnn, gap_lstm, gap_transformer]

        first_derivative_gap = np.diff(gap_mean)
        second_derivative_gap = np.diff(first_derivative_gap)
        # third_derivative_gap = np.diff(second_derivative_gap)

        # SOLO CNN
        predicted_sequence = plot_predicted_transition_shadow(real_times, models, second_derivative_gap, predictions, sample, axes[1])
        ground_truth_sequence = grouping_segments(true_labels, delay=0)
        print("predicted_sequence")
        print(predicted_sequence)
        print("ground_truth_sequence")
        print(ground_truth_sequence)
        # real_transitions, sample_accuracy = cnn_solo_results(real_times, entropies, gaps, predictions, sample, true_labels, axes[1])
        real_transitions = [(start, end-start) for label, start, end in predicted_sequence if label == 4]

        expected_transitions = plot_true_shadow(real_times, true_labels, axes[0])
        axes[0].set_ylabel("Ground\nTruth")
        axes[0].set_title('Dataset 3: sample ' + str(sample+1) + ' / 9')
        exp, corr, ghost, extra_bad, sdel, sdur = computing_transient_metrics(expected_transitions, real_transitions)
        total_expected_transitions += exp
        total_corrected_transitions += corr
        total_ghost_transitions += ghost
        total_extra_bad_transitions += extra_bad
        total_delay += sdel
        total_duration += sdur

        exit(0)
        # accuracy += sample_accuracy
        axes[1].set_ylabel("Predicted\nSequence")

        axes[2].plot(plot_times[2:], second_derivative_gap, color="blue", linewidth=2, label="GAP derivative")
        # axes[2].plot(plot_times[3:], third_derivative_gap, color="blue", linewidth=2, label="GAP derivative")
        # for i in range(1, n_samples):
        #     axes[2].axvline(x=50 * i, linestyle="--")
        axes[2].set_ylabel("GAP derivative", fontsize=12, fontweight="bold")
        axes[2].legend()

        axes[3].plot(plot_times, gap_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        # axes[3].plot(plot_times, gap_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        # axes[3].plot(plot_times, gap_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        axes[3].plot(plot_times, gap_mean, color="blue", linewidth=2, label="Mean")
        # for i in range(1, n_samples):
        #     axes[3].axvline(x=50 * i, linestyle="--")
        axes[3].set_ylabel("GAP", fontsize=12, fontweight="bold")
        axes[3].legend()

        axes[4].plot(plot_times, entropy_cnn, color="green", linewidth=2, label="CNN", alpha=0.2)
        # axes[4].plot(plot_times, entropy_lstm, color="purple", linewidth=2, label="LSTM", alpha=0.2)
        # axes[4].plot(plot_times, entropy_transformer, color="orange", linewidth=2, label="Transformer", alpha=0.2)
        axes[4].plot(plot_times, entropy_mean, color="blue", linewidth=2, label="Mean")
        # for i in range(1, n_samples):
        #     axes[4].axvline(x=50 * i, linestyle="--")
        axes[4].set_ylabel("ENTROPY", fontsize=12, fontweight="bold")
        axes[4].legend()
        plt.xlabel("Timesteps")
        plt.tight_layout()
        plt.show()
        # plt.savefig("dataset3_pred/sample_"+str(sample)+".png", bbox_inches='tight')


    ratio_ghost = total_ghost_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    ratio_extra_bad = total_extra_bad_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    mean_duration = total_duration / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_delay = total_delay / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_accuracy = accuracy / len(y_labels)
    print(f"Ratio of ghost transitions / expected: {ratio_ghost:.3f}")
    print(f"Ratio of extra bad transitions / expected: {ratio_extra_bad:.3f}")
    print(f"Mean duration of correct real transitions: {mean_duration:.2f}")
    print(f"Mean delay between expected and real correct transitions: {mean_delay:.2f}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}")