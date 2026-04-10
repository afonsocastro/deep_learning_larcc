#!/usr/bin/env python3

from numpy import load
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import statistics
from config.definitions import ROOT_DIR
from utils import grouping_segments
from src.scenario3.streaming_decision_engine import StreamingDecisionEngine



def compute_steady_metrics(gt, pred):
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


def computing_transient_metrics (gt_trans, predicted_trans):
    """
    Compute transient metrics:

    Parameters:
    - gt_trans: list of timesteps where primitive was asked to change -- ground truth
    - predicted_trans: list of tuples (initial timestep, duration) -- predicted transitions

    Returns:
    - total_expected, total_correct, ghost_transitions, total_extra_bad, sum_delay, sum_duration
    """
    correct_real_transitions = []
    ghost_transitions = 0
    extra_bad_transitions = []

    for i, expected_time in enumerate(gt_trans):
        start = expected_time
        matching_real_transitions = [(t, d) for (t, d) in predicted_trans if start - 150 < t < start + 200]

        if matching_real_transitions:
            # First one is correct
            correct_real_transitions.append(matching_real_transitions[0])
            predicted_trans.remove(matching_real_transitions[0])
            # The rest are extra
            if len(matching_real_transitions) > 1:
                for i in range(1, len(matching_real_transitions)):
                    extra_bad_transitions.append(matching_real_transitions[i])
                    predicted_trans.remove(matching_real_transitions[i])
        else:
            ghost_transitions += 1

    if predicted_trans:
        extra_bad_transitions.extend(predicted_trans[:])

    total_expected = len(gt_trans)
    total_correct = len(correct_real_transitions)
    total_extra_bad = len(extra_bad_transitions)

    sum_duration = sum(d for _, d in correct_real_transitions)
    sum_delay = sum(rt - gt_trans[i] for i, (rt, _) in enumerate(correct_real_transitions))

    return total_expected, total_correct, ghost_transitions, total_extra_bad, sum_delay, sum_duration


if __name__ == '__main__':
    time_steps = 1500
    sliding_window = 20
    t = PrettyTable(['Sample', 'Accuracy', 'Precision', 'Recall', 'F1', 'False Negatives' , 'False Positives', 'True Positives', 'Ground Truth', 'Delay', 'Duration'])
    final_table_list = []
    predictions = {"cnn": load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1.npy"),
                   "transformer": load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1.npy")}
    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    y_labels = y_data[:, :, -1]

    total_expected_transitions = 0
    total_corrected_transitions = 0
    total_ghost_transitions = 0
    total_extra_bad_transitions = 0
    total_delay = 0
    total_duration = 0
    total_accuracy = 0

    for sample in range(0, len(y_labels)):
        print("\n----------------------------------------------")
        print("sample: " +str(sample)+"/"+str(len(y_labels)-1))

        true_labels = y_labels[sample]
        ground_truth_sequence = grouping_segments(true_labels, delay=0)
        print("\nground_truth_sequence")
        print(ground_truth_sequence)

        outptus = {"cnn": None, "transformer": None}
        sde = StreamingDecisionEngine()
        for t_idx, ts in enumerate(np.array([i for i in range(0, time_steps - sliding_window + 1)])):
            outptus["cnn"] = predictions["cnn"][sample][ts]
            outptus["transformer"] = predictions["transformer"][sample][ts]
            sde.predict_haptic_sequence(outptus, ts)
        predicted_sequence = grouping_segments(sde.final_sequence, delay=sliding_window - 1)
        print("\npredicted_sequence")
        print(predicted_sequence)
        print("---------------------------------------------")

        accuracy, precision, recall, f1 = compute_steady_metrics(ground_truth_sequence, predicted_sequence)
        # predicted_transitions = [(start, end-start) for label, start, end in predicted_sequence if label == 4]
        predicted_transitions = [(start, end - start) for i, (label, start, end) in enumerate(predicted_sequence) if
                                 label == 4 and (i != 0)]
        print("predicted_transitions")
        print(predicted_transitions)

        ground_truth_transitions = [start for _, start, _ in ground_truth_sequence[1:]]
        exp, corr, ghost, extra_bad, sdel, sdur = computing_transient_metrics(ground_truth_transitions, predicted_transitions)

        total_expected_transitions += exp
        total_corrected_transitions += corr
        total_ghost_transitions += ghost
        total_extra_bad_transitions += extra_bad
        total_delay += sdel
        total_duration += sdur
        total_accuracy += accuracy

        if corr != 0:
            final_table_list.append([round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), ghost, extra_bad, corr, exp, round(sdel/corr, 3), round(sdur/corr, 3) ])
            t.add_row([sample, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), ghost, extra_bad, corr, exp, round(sdel/corr, 3), round(sdur/corr, 3) ])

    means = [round(statistics.mean(col), 3) for col in zip(*final_table_list)]
    t.add_row(["TOTAL", means[0], means[1], means[2], means[3], means[4], means[5], means[6], means[7], means[8], means[9]])
    print("\n")
    print("\n")
    print("+--------+---------------------------------------+---------------------------------------------------------------------------------------+")
    print("|        |                 Steady                |                                        Transient                                      |")
    print(t)
    ratio_ghost = total_ghost_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    ratio_extra_bad = total_extra_bad_transitions / total_expected_transitions if total_expected_transitions > 0 else float('inf')
    mean_duration = total_duration / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_delay = total_delay / total_corrected_transitions if total_corrected_transitions > 0 else 0
    mean_accuracy = total_accuracy / len(y_labels)
    print(f"\nRatio of False Negative transitions / Ground Truth transitions: {ratio_ghost:.3f}")
    print(f"Ratio of False Positive transitions / Ground Truth transitions: {ratio_extra_bad:.3f}")
    print(f"Mean duration of True Positive transitions: {mean_duration:.2f}")
    print(f"Mean delay between Ground Truth and True Positive transitions: {mean_delay:.2f}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print("\n")