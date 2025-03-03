#!/usr/bin/env python3

import numpy as np
import json

if __name__ == '__main__':
    data = np.load("../user_splitted_raw_data/Ine_learning_data_8.npy")
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0

    for i in range(0,121):

        if data[i, -1] == 0:
            count_0 += 1
        elif data[i, -1] == 1:
            count_1 += 1
        elif data[i, -1] == 2:
            count_2 += 1
        elif data[i, -1] == 3:
            count_3 += 1

    print("\ncount_0:")
    print(count_0)

    print("\ncount_1:")
    print(count_1)

    print("\ncount_2:")
    print(count_2)

    print("\ncount_3:")
    print(count_3)