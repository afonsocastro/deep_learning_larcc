#!/usr/bin/env python3

import json
import os
import numpy as np

def sample_shortener(array, old_measurements, store_config, new_measurements):
    array_shortened = np.empty((0, int(new_measurements * len(store_config["data"]))))

    classification = []
    for vector in array:
        data_array = np.reshape(vector[:-1], (old_measurements, int(len(vector) / old_measurements)))
        idx_start = 0
        idx_end = int(new_measurements)

        c = vector[-1]

        # while True:
        #     if idx_end > measurements:
        #         break
        #
        #     data_array[idx_start:idx_end, 0] = data_array[idx_start:idx_end, 0] - data_array[idx_start:idx_end, 0][0]

        vector_shortened = np.reshape(data_array[idx_start:idx_end, :], (1, array_shortened.shape[1]))
        array_shortened = np.append(array_shortened, vector_shortened, axis=0)
        classification.append(c)
        idx_start += int(new_measurements)
        idx_end += int(new_measurements)

    array_shortened = np.append(array_shortened, np.reshape([classification], (-1, 1)), axis=1)
    return array_shortened


if __name__ == '__main__':

    f = open('../config/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    path = '../user_splitted_raw_data/'
    f = open('../config/data_storage_config.json')
    storage_config = json.load(f)
    f.close()
    time_idx = 1
    f = open('../config/training_config_time_'+str(time_idx)+'.json')
    training_config = json.load(f)
    f.close()

    measurements = int(storage_config["rate"] * storage_config["time"])
    data_shortened = np.empty((0, 0))
    training_data = np.empty((0, 0))
    test_data = np.empty((0, 0))
    raw_training_data = np.empty((0, 651))
    raw_test_data = np.empty((0, 651))

    files = os.listdir(path)
    for file in files:
        new_array = np.load(path + file)
        for user in training_config["training_users"]:
            number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
            if user == number:
                print("training data: " + str(user))
                print("file: " + file)
                raw_training_data = np.append(raw_training_data, new_array, axis=0)

        for user in training_config["test_users"]:
            number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
            if user == number:
                print("test data: " + str(user))
                print("file: " + file)
                raw_test_data = np.append(raw_test_data, new_array, axis=0)

    if storage_config["time"] == training_config["time"]:
        train_data_shortened = raw_training_data
        test_data_shortened = raw_test_data
    else:
        new_measurements = storage_config["rate"] * training_config["time"]
        train_data_shortened = sample_shortener(raw_training_data, measurements, storage_config, new_measurements)
        test_data_shortened = sample_shortener(raw_test_data, measurements, storage_config, new_measurements)
        print("Time truncation complete...")
        measurements = int(new_measurements)

    np.random.shuffle(test_data_shortened)
    np.random.shuffle(train_data_shortened)
    print("Shuffle complete...")

    x_train = train_data_shortened[:, :-1]
    y_train = train_data_shortened[:, -1]
    x_test = test_data_shortened[:, :-1]
    y_test = test_data_shortened[:, -1]

    array_norm = np.empty((0, x_train.shape[1]))

    for vector in x_train:
        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        data_array_norm = np.empty((data_array.shape[0], 0))

        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

        vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

        array_norm = np.append(array_norm, vector_data_norm, axis=0)

    array_norm = np.append(array_norm, np.reshape([y_train], (-1, 1)), axis=1)
    np.save("../data1/global_normalized_train_data_"+ str(time_idx) +"00ms.npy", array_norm)

    array_norm = np.empty((0, x_test.shape[1]))

    for vector in x_test:
        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        data_array_norm = np.empty((data_array.shape[0], 0))

        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

        vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

        array_norm = np.append(array_norm, vector_data_norm, axis=0)

    array_norm = np.append(array_norm, np.reshape([y_test], (-1, 1)), axis=1)
    np.save("../data1/normalized_test_data_"+ str(time_idx) +"00ms.npy", array_norm)
