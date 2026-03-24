#!/usr/bin/env python3

import json
import numpy as np
import glob, os
from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.utils import NumpyArrayEncoder, plot_confusion_matrix_percentage, values_contabilization, \
    group_classification, mean_calc, filling_metrics_table_n, filling_metrics_table, metrics_calc, filling_table



if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    metrics_per_time_window = []

    with open("x_time_window_train_test_5.json", "r") as read_file:
        main_dict = json.load(read_file)

    time_window = main_dict["time_window"]
    training_test_list = main_dict["train_test_list"]
    n_times = len(training_test_list)
    epochs = len(training_test_list[0]["training"]["loss"])

    print("n_times")
    print(n_times)

    pull = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    push = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    shake = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    twist = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}

    for n_test in range(0, n_times):
        # -------------------------------------------------------------------------------------------------------------
        # OUTPUT CONFIDENCES-----------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        dt = training_test_list[n_test]["test"]
        group_classification(origin_dict=dt["pull"], dest_dict=pull)
        group_classification(origin_dict=dt["push"], dest_dict=push)
        group_classification(origin_dict=dt["shake"], dest_dict=shake)
        group_classification(origin_dict=dt["twist"], dest_dict=twist)

    # -------------------------------------------------------------------------------------------------------------
    # METRICS-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    metrics_pull = {"accuracy": [], "recall": [], "precision": [], "f1": []}
    metrics_push = {"accuracy": [], "recall": [], "precision": [], "f1": []}
    metrics_shake = {"accuracy": [], "recall": [], "precision": [], "f1": []}
    metrics_twist = {"accuracy": [], "recall": [], "precision": [], "f1": []}

    for l in range(0, len(pull["true_positive"])):
        metrics_calc(pull, metrics_pull, l)
        metrics_calc(push, metrics_push, l)
        metrics_calc(shake, metrics_shake, l)
        metrics_calc(twist, metrics_twist, l)

    data_metrics = filling_metrics_table(pull_metrics=metrics_pull, push_metrics=metrics_push,
                                         shake_metrics=metrics_shake, twist_metrics=metrics_twist)

    metrics_per_time_window.append({"time_window": time_window, "metrics": data_metrics})

    data_metrics_n = filling_metrics_table_n(pull_metrics=metrics_pull, push_metrics=metrics_push,
                                             shake_metrics=metrics_shake, twist_metrics=metrics_twist, n=0)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data_metrics_n, title='Metrics for one test', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data_metrics, title='Mean Metrics', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output('convo_metrics_table_' + str(time_window) + '.pdf')
