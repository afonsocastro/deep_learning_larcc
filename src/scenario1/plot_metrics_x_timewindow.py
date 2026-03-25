#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
# from deep_learning_larcc.config.PDF import PDF
from deep_learning_larcc.config.PDF2 import PDF

if __name__ == '__main__':
    labels = ["Accuracy", "Precision", "Recall", "F1 score"]
    cnn_model = "v1_1"
    # x_times = [10, 20, 30, 40, 50]
    x_times = [20, 30, 40]
    data_dict = {}
    stats_dict = {10: {"mean": [], "std_dev": []}, 20: {"mean": [], "std_dev": []}, 30: {"mean": [], "std_dev": []},
                  40: {"mean": [], "std_dev": []}, 50: {"mean": [], "std_dev": []}}

    for x_time in x_times:
        # with open(cnn_model+"/statistical_metrics_100_dataset1_"+cnn_model+"_time_window_"+str(x_time)+"_ts.json", 'r', encoding='utf-8') as file_x:
        with open("statistical_metrics_100_dataset1_"+cnn_model+"_time_window_"+str(x_time)+"_ts.json", 'r', encoding='utf-8') as file_x:
            data_dict[x_time] = json.load(file_x)
            stats_dict[x_time]["mean"] = [data_dict[x_time]["accuracy"]["mean"], data_dict[x_time]["precision"]["mean"], data_dict[x_time]["recall"]["mean"],
                 data_dict[x_time]["f1"]["mean"]]
            stats_dict[x_time]["std_dev"] = [data_dict[x_time]["accuracy"]["std_dev"], data_dict[x_time]["precision"]["std_dev"],
                                          data_dict[x_time]["recall"]["std_dev"],
                                          data_dict[x_time]["f1"]["std_dev"]]



    x = np.arange(len(labels))  # the label locations
    # width = 0.15  # the width of the bars
    # positions = {10: x - 2*width, 20: x - width, 30: x, 40: x + width, 50: x + 2*width }
    # colors = ["#8f9bff", "#47cd4d", "#fe8281", "#e6df07", "#edac5a" ]
    width = 0.2  # the width of the bars
    positions = {20: x - width, 30: x, 40: x + width}
    colors = ["#47cd4d", "#fe8281", "#edac5a" ]

    fig, ax = plt.subplots()
    alpha = 0.5
    bars = {}
    for x_time, color in zip(x_times, colors):
        # Create translucent bars for the main values and error bars
        bars[x_time] = ax.bar(positions[x_time], stats_dict[x_time]["mean"], width, label="timewindow: "+str(x_time/100)+" s", color=color, edgecolor="white",
               linewidth=2)
        ax.errorbar(positions[x_time], stats_dict[x_time]["mean"], yerr=stats_dict[x_time]["std_dev"], fmt='none',
                    color=color, ecolor="black", elinewidth=2, capsize=5, alpha= alpha)

    # bars_v1_1 = ax.bar(x - (3*width/2), v1_1_mean, width, label='v1.1', color="#47cd4d", edgecolor="white",
    #        linewidth=2)  # Translucent bars
    # ax.errorbar(x - (3*width/2), v1_1_mean, yerr=v1_1_std_dev, fmt='none',
    #             color="#47cd4d", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)
    #
    # bars_v1_2 = ax.bar(x - (width/2), v1_2_mean, width, label='v1.2', color="#fe8281", edgecolor="white",
    #        linewidth=2)
    # ax.errorbar(x - (width/2), v1_2_mean, yerr=v1_2_std_dev, fmt='none',
    #             color="#fe8281", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)
    #
    # bars_v1_3 = ax.bar(x + (width/2), v1_3_mean, width, label='v1.3', color="#e6df07", edgecolor="white",
    #        linewidth=2)
    # ax.errorbar(x + (width/2), v1_3_mean, yerr=v1_3_std_dev, fmt='none',
    #             color="#e6df07", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)
    #
    # bars_v1_4 = ax.bar(x + (3*width/2), v1_4_mean, width, label='v1.4', color="#b46eff", edgecolor="white",
    #                    linewidth=2)
    # ax.errorbar(x + (3*width/2), v1_4_mean, yerr=v1_4_std_dev, fmt='none',
    #             color="#b46eff", ecolor="black", elinewidth=2, capsize=5, alpha=alpha)
    #
    # bars_v1_5 = ax.bar(x + (5*width/2), v1_5_mean, width, label='v1.5', color="#cc9600", edgecolor="white",
    #                    linewidth=2)
    # ax.errorbar(x + (5*width/2), v1_5_mean, yerr=v1_5_std_dev, fmt='none',
    #             color="#cc9600", ecolor="black", elinewidth=2, capsize=5, alpha=alpha)

    # Add the values on top of each main bar (not on error bars)
    for bars, values in zip([bars[x_time] for x_time in x_times], [stats_dict[x_time]["mean"] for x_time in x_times]):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value}',
                    ha='center', va='bottom', fontsize=10, color='black')

    # Add custom x-axis tick labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Mean & Standard Deviation - Convolutional v1.1 - 100 simulations (train + test dataset1)", fontsize=14,
                 fontweight='bold')

    # Show the plot
    fig.tight_layout()
    delta = 0.03
    plt.ylim([min(stats_dict[10]["mean"] + stats_dict[20]["mean"] + stats_dict[30]["mean"] + stats_dict[40]["mean"] + stats_dict[50]["mean"]) - delta,
              max(stats_dict[10]["mean"] + stats_dict[20]["mean"] + stats_dict[30]["mean"] + stats_dict[40]["mean"] + stats_dict[50]["mean"]) + delta])
    plt.show()

    # Table comparing every metrics (in pdf) -------------------------------------------------------------------------
    # Interquartile Range: the smaller the better
    # Coefficient of Variation: the lower the better

    # pdf = PDF(title='Statistical Metrics - 100 simulations (train + dataset1 test)')
    # pdf.add_page()
    # pdf.set_font("Times", size=9)
    #
    # for mtrc, met_title in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
    #                         ("f1", "F1-score")]:
    #     data = [
    #         [met_title, "Interquartile Range", "Coefficient of Variation", "95% Confidence Interval", ],
    #         ["LSTM v1.0", data_v1_0[mtrc]["iqr"], data_v1_0[mtrc]["cv"], data_v1_0[mtrc][
    #             "95_confidence_interval"], ],
    #         ["LSTM v1.1", data_v1_1[mtrc]["iqr"], data_v1_1[mtrc]["cv"], data_v1_1[mtrc][
    #             "95_confidence_interval"], ],
    #         ["LSTM v1.2", data_v1_2[mtrc]["iqr"], data_v1_2[mtrc]["cv"], data_v1_2[mtrc][
    #             "95_confidence_interval"], ],
    #         ["LSTM v1.3", data_v1_3[mtrc]["iqr"], data_v1_3[mtrc]["cv"], data_v1_3[mtrc][
    #             "95_confidence_interval"], ],
    #         ["LSTM v1.4", data_v1_4[mtrc]["iqr"], data_v1_4[mtrc]["cv"], data_v1_4[mtrc][
    #             "95_confidence_interval"], ],
    #         ["LSTM v1.5", data_v1_5[mtrc]["iqr"], data_v1_5[mtrc]["cv"], data_v1_5[mtrc][
    #             "95_confidence_interval"], ]
    #     ]
    #     pdf.create_table(table_data=data)
    #     pdf.ln()
    #     pdf.ln()
    # pdf.output('lstms_v1_statistical_metrics.pdf')