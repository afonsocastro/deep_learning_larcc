#!/usr/bin/env python3

from numpy import load
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

# Function to convert dictionary values into hashable types
def make_hashable(d):
    return frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())

def identify_same_dictionary_in_list(list_of_dicts):
    # Flatten all dictionaries and convert them into hashable types
    all_dicts = [make_hashable(d) for lst in list_of_dicts for d in lst]
    counts = Counter(all_dicts)

    # Find elements appearing in all lists
    common_dicts = [dict(k) for k, v in counts.items() if v == len(list_of_dicts)]
    return common_dicts


if __name__ == '__main__':
    versions = ["v1_0", "v1_2", "v1_1", "v1_3", "v1_4", "v1_5"]
    datas = {v: {} for v in versions}
    for version in versions:
        with open(f"{version}/data2_metrics_transformer_{version}.json", 'r', encoding='utf-8') as file:
            datas[f"{version}"] = json.load(file)

    for key, value in datas.items():
        print(f"\n {key} reactivity: {value['reactivity']}")
        print(f"Pos Transition Miss Classification: {len(value['pos_transition_miss_classification'])} : {(len(value['pos_transition_miss_classification']) / 565) * 100}")
        print(f"Pre Transition same as 2nd action: {len(value['prev_predict_is_2nd_action'])} : {(len(value['prev_predict_is_2nd_action']) / 565) * 100}")

    lists_post_transition = []
    lists_pre_transition = []
    for key, value in datas.items():
        lists_post_transition.append(value["pos_transition_miss_classification"])
        lists_pre_transition.append(value["prev_predict_is_2nd_action"])

    common_dicts_pt = identify_same_dictionary_in_list(lists_post_transition)
    print("\ncommon values for Post Transition Miss Classification:")
    print(common_dicts_pt)

    pt_common_dicts = identify_same_dictionary_in_list(lists_pre_transition)
    print("\ncommon values for Pre Transition w/ same action as 2nd GT:")
    print(pt_common_dicts)

    miss_count = {(0, 0): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (0, 1): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (0, 2): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (0, 3): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (1, 0): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (1, 1): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (1, 2): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (1, 3): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (2, 0): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (2, 1): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (2, 2): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (2, 3): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (3, 0): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (3, 1): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (3, 2): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0},
                   (3, 3): {"v1_0": 0, "v1_2": 0, "v1_1": 0, "v1_3": 0, "v1_4": 0, "v1_5": 0}}

    for key, value in datas.items():
        for d in value['prev_predict_is_2nd_action']:
            for x in range(4):
                for y in range(4):
                    if d["gt"] == [x,y]:
                        miss_count[tuple(d["gt"])][key] += 1

    labels = ["V1.0", "V1.1", "V1.2", "V1.3", "V1.4", "V1.5"]
    colors = ["#8f9bff", "#47cd4d", "#fe8281", "#e6df07", "#b46eff", "#cc9600"]
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            values = [miss_count[(i,j)][version] for version in versions]
            axs[i, j].bar(labels, values , color=colors, width=1)
            axs[i, j].set_xticklabels([])
            axs[i, j].set_ylim([0,17])
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    # plt.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()

