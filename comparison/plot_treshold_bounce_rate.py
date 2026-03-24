#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open("rejected_samples_percentage_histograms_treshold_dataset1.json", 'r', encoding='utf-8') as file:
        data_bounce_rate = json.load(file)

    # Plotting
    plt.figure(figsize=(10, 6))

    for model_name, values in data_bounce_rate.items():
        x = [point[0] for point in values]
        y = [point[1] for point in values]
        plt.plot(x, y, marker='o', label=model_name)

    plt.title("Bounce Rate per Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Bounce Rate %")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()