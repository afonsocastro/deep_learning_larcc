#!/usr/bin/env python3

import keras
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.utils import plot_confusion_matrix_percentage, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay
from progressbar import progressbar
from deep_learning_larcc.utils import NumpyArrayEncoder
import numpy as np
from deep_learning_larcc.config.definitions import ROOT_DIR

if __name__ == '__main__':
    with open("metrics_histograms_treshold_100_dataset1.json", 'r', encoding='utf-8') as file:
        data_histograms = json.load(file)

    metrics = ["accuracy", "precision", "recall", "f1"]

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Classification metrics / threshold")
    delta = 0.01

    # Iterate over metrics and corresponding subplot axes
    for ax, metric in zip(axes.flat, metrics):
        x_cnn, y_cnn = zip(*data_histograms["cnn"][metric])
        x_lstm, y_lstm = zip(*data_histograms["lstm"][metric])
        x_transformer, y_transformer = zip(*data_histograms["transformer"][metric])

        ax.plot(x_cnn, y_cnn, marker='o', label="CNN")
        ax.plot(x_lstm, y_lstm, marker='o', label="LSTM")
        ax.plot(x_transformer, y_transformer, marker='o', label="Transformer")

        # Set y-axis limits
        y_min = min(min(y_cnn), min(y_lstm), min(y_transformer))
        y_max = max(max(y_cnn), max(y_lstm), max(y_transformer))
        ax.set_ylim(y_min - delta, y_max + delta)  # Add some padding

        ax.set_xlabel("Threshold")
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.show()