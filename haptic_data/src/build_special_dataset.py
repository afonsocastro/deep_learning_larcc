#!/usr/bin/env python3

import numpy as np
import json
import random

if __name__ == '__main__':
    min_size = 4
    max_size = 10
    group_size = 10
    dataset1_test = np.load("../data1/global_normalized_test_data_500ms.npy")
    np.random.shuffle(dataset1_test)
    y_test = dataset1_test[:, -1]

    print("before balanced generation....\n")
    x_test_new = np.empty((0, 6500))
    y_test_new = []

    groups = []

    while len(y_test) > 0:
        # group_size = random.randint(min_size, min(max_size, len(y_test)))
        new_array = np.concatenate([dataset1_test[idx, :-1] for idx in range(group_size)])
        new_array = np.reshape(new_array, (1, new_array.shape[0]))
        x_test_new = np.append(x_test_new, new_array , axis=0)
        y_test_new.append([y_test[idx] for idx in range(group_size)])
        dataset1_test = dataset1_test[group_size:, :]  # Remove used elements
        y_test = y_test[group_size:]  # Remove used elements

    x_test_new = np.reshape(x_test_new, (113, 500, 13))
    np.save("../data3/x_test_data.npy", x_test_new)
    np.save("../data3/y_test_data.npy", y_test_new)