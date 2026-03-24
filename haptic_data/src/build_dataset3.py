#!/usr/bin/env python3

import json
import os
import numpy as np
import random
import keras


if __name__ == '__main__':
    test_users = [5, 6, 7, 8, 9, 10, 11]
    f = open('clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    data_arrays = {5: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   6: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   7: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   8: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   9: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   10: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))},
                   11: {0: np.empty((0, 651)), 1: np.empty((0, 651)), 2: np.empty((0, 651)), 3: np.empty((0, 651))}}

    # CREATING THE POTS--------------------------------------------------------------------------------------------------
    path = '../user_splitted_raw_data/'
    files = os.listdir(path)
    for file in files:
        new_array = np.load(path + file)
        for user in test_users:
            number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
            if user == number:
                print("test data: " + str(user))
                print("file: " + file)
                # Normalization
                x_test = new_array[:, :-1]
                y_test = new_array[:, -1]
                array_norm = np.empty((0, 650))
                for vector, y_value in zip(x_test, y_test):

                    data_array = np.reshape(vector, (50, int(len(vector) / 50)))
                    data_array_norm = np.empty((data_array.shape[0], 0))

                    data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
                    data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
                    data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
                    data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

                    vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))
                    vector_data_norm = np.append(vector_data_norm, [[y_value]], axis=1)

                    data_arrays[user][y_value] = np.append(data_arrays[user][y_value], vector_data_norm, axis=0)

    # TESTING DATASET1 to EXCLUDE "BAD" SAMPLES ------------------------------------------------------------------------
    cnn_model = "v1_1"
    lstm_model = "v1_2"
    transformer_model = "v1_1"
    model_cnn = keras.models.load_model("../../convolutional/v1/"+cnn_model+"/cnn_"+cnn_model+".keras")
    model_lstm = keras.models.load_model("../../recurrent/v1/"+lstm_model+"/lstm_"+lstm_model+".keras")
    model_transformer = keras.models.load_model("../../transformers/v1/"+transformer_model+"/transformer_"+transformer_model+".keras")

    for user in test_users:
        print("\nUser " + str(user) + ": ")
        total_count = 0
        for p in [0,1,2,3]:
            array = data_arrays[user][p]
            y_test = array[:, -1]
            x_test = array[:, :260]
            x_test = np.reshape(x_test, (array.shape[0], 20, 13, 1))
            x_test = x_test[:, :, 1:, :]
            rows_to_remove = []
            for i in range(0, len(array)):
                prediction_cnn = model_cnn.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
                prediction_lstm = model_lstm.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
                prediction_transformer = model_transformer.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
                decoded_prediction_cnn = np.argmax(prediction_cnn, axis=1, out=None)
                decoded_prediction_lstm = np.argmax(prediction_lstm, axis=1, out=None)
                decoded_prediction_transformer = np.argmax(prediction_transformer, axis=1, out=None)
                true = y_test[i]
                if true != decoded_prediction_cnn or true != decoded_prediction_lstm or true != decoded_prediction_transformer:
                    rows_to_remove.append(i)
            print(str(p)+" , rows_to_remove:")
            print(rows_to_remove)
            data_arrays[user][p] = np.delete(data_arrays[user][p], rows_to_remove, axis=0)


    # CONCATENATING 7 ACTIONS AT A TIME (139 SAMPLES) ------------------------------------------------------------------------

    X_samples = []
    y_samples = []
    used_indices = {user: {t: set() for t in range(4)} for user in data_arrays}

    for _ in range(139):
        x_parts = []
        y_part = []
        prev_user = None
        prev_type = None

        for _ in range(7):
            valid_choices = []

            for user in data_arrays:
                for t in range(4):
                    available_indices = [
                        i for i in range(len(data_arrays[user][t]))
                        if i not in used_indices[user][t]
                    ]
                    if not available_indices:
                        continue
                    if prev_type is not None and t == prev_type and user != prev_user:
                        continue
                    valid_choices.append((user, t, random.choice(available_indices)))

            if not valid_choices:
                raise RuntimeError("Ran out of valid elements to sample from.")

            user, t, idx = random.choice(valid_choices)
            element = data_arrays[user][t][idx]  # Shape: (1, 651)
            used_indices[user][t].add(idx)

            # Split input and label
            x = element[:-1].reshape(50, 13)  # (50, 13)
            y = int(element[-1])  # scalar label
            x_parts.append(x)
            y_part.append(y)

            prev_user = user
            prev_type = t

        # Combine 7 x (50, 13) → (350, 13)
        x_sample = np.vstack(x_parts)
        X_samples.append(x_sample)
        y_samples.append(y_part)

    X_samples = np.stack(X_samples)  # (139, 350, 13)
    y_samples = np.array(y_samples)  # (139, 7)

    print("X_samples")
    print(type(X_samples))
    print(X_samples.shape)
    print("Y_samples")
    print(type(y_samples))
    print(y_samples.shape)

    for user in test_users:
        print("\nUser " + str(user) + ": ")
        total_count = 0
        for p in [0,1,2,3]:
            print(str(p)+": "+str(data_arrays[user][p].shape))
            total_count += data_arrays[user][p].shape[0]

        print("Total: "+str(total_count))

    np.save("x_test_data.npy", X_samples)
    np.save("y_test_data.npy", y_samples)
