#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
# from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.config.PDF2 import PDF

if __name__ == '__main__':
    labels = ["Accuracy", "Precision", "Recall", "F1 score"]
    cnn_model = "v1_1"
    lstm_model = "v1_2"
    transformer_model = "v1_1"
    with open("../convolutional/v1/"+cnn_model+"/statistical_metrics_100_dataset1_"+cnn_model+".json", 'r', encoding='utf-8') as file_cnn:
        data_cnn = json.load(file_cnn)
    with open("../recurrent/v1/"+lstm_model+"/statistical_metrics_100_dataset1_"+lstm_model+".json", 'r', encoding='utf-8') as file_lstm:
        data_ltm = json.load(file_lstm)
    with open("../transformers/v1/"+transformer_model+"/statistical_metrics_100_dataset1_"+transformer_model+".json", 'r', encoding='utf-8') as file_transformer:
        data_transformer = json.load(file_transformer)

    # Table comparing every metrics (in pdf) -------------------------------------------------------------------------
    # Interquartile Range: the smaller the better
    # Coefficient of Variation: the lower the better

    pdf = PDF(title='Statistical Metrics - 100 simulations (train + dataset1 test)')
    pdf.add_page()
    pdf.set_font("Times", size=9)

    for mtrc, met_title in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                            ("f1", "F1-score")]:
        data = [
            [met_title, "Mean", "Standard Deviation", "Interquartile Range", "Coefficient of Variation", "95% Confidence Interval", ],
            ["CNN (v1.1)", data_cnn[mtrc]["mean"], data_cnn[mtrc]["std_dev"], data_cnn[mtrc]["iqr"],
             data_cnn[mtrc]["cv"], data_cnn[mtrc]["95_confidence_interval"], ],
            ["LSTM (v1.2)", data_ltm[mtrc]["mean"], data_ltm[mtrc]["std_dev"], data_ltm[mtrc]["iqr"],
             data_ltm[mtrc]["cv"], data_ltm[mtrc]["95_confidence_interval"], ],
            ["Transformer (v1.1)", data_transformer[mtrc]["mean"], data_transformer[mtrc]["std_dev"], data_transformer[mtrc]["iqr"],
             data_transformer[mtrc]["cv"], data_transformer[mtrc]["95_confidence_interval"], ]
        ]
        pdf.create_table(table_data=data)
        pdf.ln()
        pdf.ln()
    pdf.output('models_comparison_statistical_metrics_dataset1.pdf')

