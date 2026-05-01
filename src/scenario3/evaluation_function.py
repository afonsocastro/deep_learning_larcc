#!/usr/bin/env python3

import numpy as np
from numpy import load
from config.definitions import ROOT_DIR
from utils import grouping_segments
from src.scenario3.streaming_decision_engine import StreamingDecisionEngine
from sequence_metrics_evaluation import is_sequence_correct, computing_transient_metrics


def evaluate_model(params):
    """
    Parameters: dict with SDE parameters
    Returns: dict with metrics
    """

    time_steps = 1500
    sliding_window = 20
    predictions = {
        "cnn": load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1_filtered_2.npy"),
        "transformer": load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1_filtered_2.npy")
    }
    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_filtered_2.npy")
    y_labels = y_data[:, :, -1]

    correct_sequences = []
    sequences_delay = []
    sequences_duration = []
    false_positives_transitions = []

    for sequence in range(len(y_labels)):

        true_labels = y_labels[sequence]
        gt_seq = grouping_segments(true_labels, delay=0)

        # ---- SDE com parâmetros ----
        sde = StreamingDecisionEngine(**params)

        outputs = {"cnn": None, "transformer": None}
        for ts in range(0, time_steps - sliding_window + 1):
            outputs["cnn"] = predictions["cnn"][sequence][ts]
            outputs["transformer"] = predictions["transformer"][sequence][ts]
            sde.predict_haptic_sequence(outputs, ts)

        pred_seq = grouping_segments(sde.final_sequence, delay=sliding_window - 1)

        # ---- steady metrics ----
        sequence_correct = is_sequence_correct(gt_seq, pred_seq)

        # ---- transitions ----
        pred_trans = [(start, end - start) for i, (label, start, end) in enumerate(pred_seq) if label == 4 and i != 0]
        gt_trans = [start for _, start, _ in gt_seq[1:]]

        exp, corr, fn, fp, sdel, sdur = computing_transient_metrics(gt_trans, pred_trans)

        correct_sequences.append(sequence_correct)

        if sequence_correct == 1:
            false_positives_transitions.append(fp)
            sequences_delay.append(sdel / (corr + fn))
            sequences_duration.append(sdur / (corr + fn))


    results = {"accuracy": np.mean(correct_sequences),
               "fp_transitions": np.mean(false_positives_transitions),
               "mean_delay": np.mean(sequences_delay),
               "mean_duration": np.mean(sequences_duration)}

    return results