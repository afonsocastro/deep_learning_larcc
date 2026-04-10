#!/usr/bin/env python3

import numpy as np
from numpy import load
from config.definitions import ROOT_DIR
from utils import grouping_segments
from src.scenario3.streaming_decision_engine import StreamingDecisionEngine
from sequence_metrics_evaluation import compute_steady_metrics, computing_transient_metrics


def evaluate_model(params):
    """
    Parameters: dict with SDE parameters
    Returns: dict with metrics
    """

    time_steps = 1500
    sliding_window = 20
    predictions = {
        "cnn": load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1.npy"),
        "transformer": load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1.npy")
    }
    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    y_labels = y_data[:, :, -1]

    total_expected = 0
    total_correct = 0
    total_fn = 0
    total_fp = 0
    total_delay = 0
    total_duration = 0
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    for sample in range(len(y_labels)):

        true_labels = y_labels[sample]
        gt_seq = grouping_segments(true_labels, delay=0)

        # ---- SDE com parâmetros ----
        sde = StreamingDecisionEngine(**params)

        outputs = {"cnn": None, "transformer": None}
        for ts in range(0, time_steps - sliding_window + 1):
            outputs["cnn"] = predictions["cnn"][sample][ts]
            outputs["transformer"] = predictions["transformer"][sample][ts]
            sde.predict_haptic_sequence(outputs, ts)

        pred_seq = grouping_segments(sde.final_sequence, delay=sliding_window - 1)

        # ---- steady metrics ----
        acc, prec, rec, f1 = compute_steady_metrics(gt_seq, pred_seq)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        # ---- transitions ----
        pred_trans = [(start, end - start) for i, (label, start, end) in enumerate(pred_seq) if label == 4 and i != 0]
        gt_trans = [start for _, start, _ in gt_seq[1:]]

        exp, corr, fn, fp, sdel, sdur = computing_transient_metrics(gt_trans, pred_trans)
        total_expected += exp
        total_correct += corr
        total_fn += fn
        total_fp += fp
        total_delay += sdel
        total_duration += sdur


    results = {"accuracy": np.mean(acc_list), "precision": np.mean(prec_list), "recall": np.mean(rec_list),
               "f1": np.mean(f1_list),
               "transition_precision": total_correct / (total_correct + total_fp) if (total_correct + total_fp) > 0 else 0,
               "transition_recall": total_correct / total_expected if total_expected > 0 else 0,
               "mean_delay": total_delay / total_correct if total_correct > 0 else 0,
               "mean_duration": total_duration / total_correct if total_correct > 0 else 0,
               "fn_ratio": total_fn / total_expected if total_expected > 0 else 0,
               "fp_ratio": total_fp / total_expected if total_expected > 0 else 0}

    return results