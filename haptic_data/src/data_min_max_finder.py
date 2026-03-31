#!/usr/bin/env python3

import numpy as np
from config.definitions import ROOT_DIR
import json
import glob
import os


if __name__ == '__main__':

    f = open(ROOT_DIR + '/haptic_data/config/training_config.json')
    training_config = json.load(f)
    f.close()

    # measurements = int(training_config["rate"] * training_config["time"])
    measurements = 50

    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Ali_learning_data_9.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Ine_learning_data_8.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Joe_learning_data_11.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Luc_learning_data_7.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Maf_learning_data_6.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Mar_learning_data_1.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Mig_learning_data_10.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/P_learning_data_2.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Ro_learning_data_3.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/V_learning_data_4.npy")
    # experiment_data = np.load(ROOT_DIR + "/haptic_data/user_splitted_raw_data/Ru_learning_data_5.npy")

    clusters_max = {"timestamp": {"max": 0, "min": 0}, "joints": {"max": 0, "min": 0},
                    "gripper_F": {"max": 0, "min": 0}, "gripper_M": {"max": 0, "min": 0}}

    folder = os.path.join(ROOT_DIR, "haptic_data/user_splitted_raw_data")

    for file_path in glob.glob(os.path.join(folder, "*.npy")):
        experiment_data = np.load(file_path)

        learning_array = experiment_data[:, :-1]

        for i, vector in enumerate(learning_array):
            data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
            idx = 0
            cont = 0
            for n in training_config["normalization_clusters"]:
                data_sub_array = data_array[:, idx:idx + n]
                idx += n

                min = data_sub_array.min()
                max = data_sub_array.max()

                if cont == 0:
                    if min < clusters_max["timestamp"]["min"]:
                        clusters_max["timestamp"]["min"] = min
                    elif max > clusters_max["timestamp"]["max"]:
                        clusters_max["timestamp"]["max"] = max
                elif cont == 1:
                    if min < clusters_max["joints"]["min"]:
                        clusters_max["joints"]["min"] = min
                    elif max > clusters_max["joints"]["max"]:
                        clusters_max["joints"]["max"] = max
                elif cont == 2:
                    if min < clusters_max["gripper_F"]["min"]:
                        clusters_max["gripper_F"]["min"] = min
                    elif max > clusters_max["gripper_F"]["max"]:
                        clusters_max["gripper_F"]["max"] = max
                elif cont == 3:
                    if min < clusters_max["gripper_M"]["min"]:
                        clusters_max["gripper_M"]["min"] = min
                    elif max > clusters_max["gripper_M"]["max"]:
                        clusters_max["gripper_M"]["max"] = max
                else:
                    print("WHAAAAAATTTTTT?????????")

                cont += 1

    print(clusters_max)
    with open("../config/clusters_max_min.json", "w") as fp:
        json.dump(clusters_max, fp)
