#!/usr/bin/env python3

import numpy as np
import json
import random


def generate_balanced_pairs_with_indices(input_list):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for num in input_list:
        counts[num] += 1
    print("counts:\n")
    print(counts)
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 3),
        (3, 0), (3, 1), (3, 2)
    ]
    pairs_count = {(0, 1): 0, (0, 2): 0, (0, 3): 0,
        (1, 0): 0, (1, 2): 0, (1, 3): 0,
        (2, 0): 0, (2, 1): 0, (2, 3): 0,
        (3, 0): 0, (3, 1): 0, (3, 2): 0}
    value_pairs = []
    index_pairs = []

    indices = {0: [], 1: [], 2: [], 3: []}
    for idx, num in enumerate(input_list):
        indices[num].append(idx)

    print("\nindices")
    print(indices)
    while len(value_pairs) < 550:
        random.shuffle(pairs)

        for pair in pairs:
            a, b = pair
            print("\nindices")
            print(indices)
            if counts[a] > 0 and counts[b] > 0 and indices[a] and indices[b]:
                # print("Created!! :D")
                idx_a = indices[a].pop()
                idx_b = indices[b].pop()
                value_pairs.append((a, b))
                index_pairs.append((idx_a, idx_b))
                counts[a] -= 1
                counts[b] -= 1
                pairs_count[pair] += 1
                break

    print("\nDoes the indices list have indices for all [0,1,2,3]?")
    print("No - Ctr+C")
    input("Yes - Enter")
    sum = counts[0] + counts[1] + counts[2] + counts[3]
    while sum != 0:
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        a, b = sorted_counts[0][0], sorted_counts[1][0]
        print("\nPair")
        print(a)
        print(b)
        idx_a = indices[a].pop()
        idx_b = indices[b].pop()
        value_pairs.append((a, b))
        index_pairs.append((idx_a, idx_b))
        counts[a] -= 1
        counts[b] -= 1
        pairs_count[(a,b)] += 1
        print("\nindices")
        print(indices)
        input("\nContinue?")
        sum = counts[0] + counts[1] + counts[2] + counts[3]
    return index_pairs, value_pairs, pairs_count


if __name__ == '__main__':
    dataset1_test = np.load("../data1/normalized_test_data_500ms.npy")
    y_test = dataset1_test[:, -1]

    print("before balanced generation....\n")
    x_index_pairs, ground_truth_pairs, pairs_count = generate_balanced_pairs_with_indices(y_test)
    x_test = np.empty((0, 1300))

    # print("\nx_index_pairs:")
    # print(x_index_pairs)

    for idx_pair in x_index_pairs:
        new_array = np.concatenate((dataset1_test[idx_pair[0], :-1], dataset1_test[idx_pair[1], :-1]))
        new_array = np.reshape(new_array, (1, new_array.shape[0]))
        print("new_array.shape")
        print(new_array.shape)
        print("x_test.shape")
        print(x_test.shape)
        x_test = np.append(x_test, new_array , axis=0)

    x_test = np.reshape(x_test, (565, 100, 13))
    np.save("../data2/x_test_data.npy", x_test)

    # y_test_final = []
    # for line in range(0, y_test.shape[0], 2):
    #     r = [int(y_test[line]), int(y_test[line + 1])]
    #     y_test_final.append(r)
    y_test_final = np.array(ground_truth_pairs)
    np.save("../data2/y_test_data.npy", ground_truth_pairs)

    print("\npairs_count:")
    print(pairs_count)

    # This returns error:
    with open("../data2/pairs_count.json", "w") as write_file:
        json.dump(pairs_count, write_file)

