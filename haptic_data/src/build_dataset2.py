#!/usr/bin/env python3

import numpy as np
import json

if __name__ == '__main__':
    dataset1_test = np.load("../data1/global_normalized_test_data_500ms.npy")

    x_test = np.reshape(dataset1_test[:, :-1], (int(dataset1_test.shape[0] / 2), 100, 13))
    y_test = dataset1_test[:, -1]

    np.save("../data2/x_test_data.npy", x_test)

    y_test_final = []
    for line in range(0, y_test.shape[0], 2):
        r = [int(y_test[line]), int(y_test[line + 1])]
        y_test_final.append(r)
    y_test_final = np.array(y_test_final)
    np.save("../data2/y_test_data.npy", y_test_final)