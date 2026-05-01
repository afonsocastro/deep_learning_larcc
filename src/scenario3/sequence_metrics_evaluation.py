#!/usr/bin/env python3

from numpy import load
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import statistics
from config.definitions import ROOT_DIR
from utils import grouping_segments
from src.scenario3.streaming_decision_engine import StreamingDecisionEngine


#
# def compute_steady_metrics(gt, pred):
#     """
#     Compute segment-level Accuracy, Precision, Recall, F1 using IoU approach.
#
#     Parameters:
#     - gt: list of tuples (label, start, end) -- ground truth (no gray)
#     - pred: list of tuples (label, start, end) -- predicted (label 4 = gray)
#
#     Returns:
#     - accuracy, precision, recall, f1
#     """
#     tp_len = 0  # True positive length
#     total_pred_len = 0
#     total_gt_len = 0
#
#     for g_label, g_start, g_end in gt:
#         total_gt_len += g_end - g_start + 1
#
#         # Find predicted segments that overlap with this gt segment
#         overlap_tp = 0
#         overlap_pred_len = 0
#
#         for p_label, p_start, p_end in pred:
#             # Skip gray areas
#             if p_label == 4:
#                 continue
#
#             # Compute overlap
#             start = max(g_start, p_start)
#             end = min(g_end, p_end)
#             if start <= end:
#                 overlap_len = end - start + 1
#                 overlap_pred_len += p_end - p_start + 1  # sum predicted length
#                 if p_label == g_label:
#                     overlap_tp += overlap_len
#
#         tp_len += overlap_tp
#         total_pred_len += overlap_pred_len
#
#     # Handle cases where there is no prediction
#     if total_pred_len == 0:
#         precision = 0.0
#     else:
#         precision = tp_len / total_pred_len
#
#     recall = tp_len / total_gt_len
#     if precision + recall == 0:
#         f1 = 0.0
#     else:
#         f1 = 2 * precision * recall / (precision + recall)
#
#     accuracy = tp_len / total_gt_len
#     return accuracy, precision, recall, f1


def is_sequence_correct(gt_sequence, pred_sequence, transition_class=4):
    """
    Returns 1 if the sequence of classes matches exactly (ignoring transitions and time).
    """

    def extract(seq):
        return [c for (c, s, e) in seq if c != transition_class]

    gt_classes = extract(gt_sequence)
    pred_classes = extract(pred_sequence)

    return int(gt_classes == pred_classes)


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
    gt_real_trans = []

    for i, expected_time in enumerate(gt_trans):
        start = expected_time
        matching_real_transitions = [(t, d) for (t, d) in predicted_trans if start - 150 < t < start + 200]

        if matching_real_transitions:
            # First one is correct
            correct_real_transitions.append(matching_real_transitions[0])
            gt_real_trans.append(expected_time)
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
    sum_delay = sum(rt - gt_real_trans[i] for i, (rt, _) in enumerate(correct_real_transitions))

    return total_expected, total_correct, ghost_transitions, total_extra_bad, sum_delay, sum_duration


if __name__ == '__main__':
    time_steps = 1500
    sliding_window = 20
    t = PrettyTable(['Sequence', 'Accuracy', 'False Negatives' , 'False Positives', 'True Positives', 'Ground Truth', 'Delay', 'Duration'])
    final_table_list = []
    predictions = {"cnn": load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1_filtered_2.npy"),
                   "transformer": load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1_filtered_2.npy")}
    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_filtered_2.npy")
    y_labels = y_data[:, :, -1]
    sequences_corrected_classified = 0
    total_sequences_numer = len(y_labels)

    for sequence in range(0, total_sequences_numer):
        # print("\n----------------------------------------------")
        # print("sequence: " +str(sequence)+"/"+str(len(y_labels)-1))

        true_labels = y_labels[sequence]
        ground_truth_sequence = grouping_segments(true_labels, delay=0)
        # print("\nground_truth_sequence")
        # print(ground_truth_sequence)

        outptus = {"cnn": None, "transformer": None}
        # 0.19290008800399622 | 1.133997856446312 | 0.513136480067444 | 0.48686351993255605 | 16
        # 0.316754541299874 | 1.1001997602480325 | 0.4028921319706519 | 0.5971078680293481 | 25
        sde = StreamingDecisionEngine(shake_threshold=0.316754541299874, entropy_threshold=1.1001997602480325,
                                      cnn_weight=0.4028921319706519, transformer_weight=0.5971078680293481,
                                      min_steady_timesteps=25)
        # sde = StreamingDecisionEngine()
        for t_idx, ts in enumerate(np.array([i for i in range(0, time_steps - sliding_window + 1)])):
            outptus["cnn"] = predictions["cnn"][sequence][ts]
            outptus["transformer"] = predictions["transformer"][sequence][ts]
            sde.predict_haptic_sequence(outptus, ts)
        predicted_sequence = grouping_segments(sde.final_sequence, delay=sliding_window - 1)
        # print("\npredicted_sequence")
        # print(predicted_sequence)
        # print("---------------------------------------------")

        sequence_correct = is_sequence_correct(ground_truth_sequence, predicted_sequence)
        predicted_transitions = [(start, end - start) for i, (label, start, end) in enumerate(predicted_sequence) if
                                 label == 4 and (i != 0)]

        ground_truth_transitions = [start for _, start, _ in ground_truth_sequence[1:]]
        exp, corr, ghost, extra_bad, sdel, sdur = computing_transient_metrics(ground_truth_transitions, predicted_transitions)

        if (corr+ghost) != 0 and sequence_correct==1 :
            sequences_corrected_classified += 1
            final_table_list.append([ghost, extra_bad, corr, exp, round(sdel/(corr+ghost), 3), round(sdur/(corr+ghost), 3) ])
            t.add_row([sequence, "✅", ghost, extra_bad, corr, exp, round(sdel/(corr+ghost), 3), round(sdur/(corr+ghost), 3) ])
        elif (corr+ghost) != 0 and sequence_correct==0:
            t.add_row(["❌ "+str(sequence)+" ❌", "❌", ghost, extra_bad, corr, exp, "❌", "❌"])

    sequence_level_accuracy = sequences_corrected_classified/total_sequences_numer
    means = [round(statistics.mean(col), 3) for col in zip(*final_table_list)]
    t.add_row(["MEAN", round(sequence_level_accuracy, 3), means[0], means[1], means[2], means[3], means[4], means[5]])
    print("\n")
    print("\n")
    print("+----------+----------+---------------------------------------------------------------------------------------+")
    print("|          |          |                                        Transient                                      |")
    print(t)
