#!/usr/bin/env python3

import json
from config.definitions import ROOT_DIR
import numpy as np


if __name__ == '__main__':

    # data = np.load(ROOT_DIR + '/data_storage/data/new_acquisition/user_splitted_data/Joe_learning_data_11.npy',
    #                mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_filtered.npy")
    cnn_data = np.load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1_filtered.npy")
    lstm_data = np.load(ROOT_DIR + "/recurrent/dataset3_results/data3_pred_lstm_v1_2_filtered.npy")
    transformer_data = np.load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1_filtered.npy")

    # remove_idx = [6, 25, 26, 31, 33, 34, 36, 37, 48, 50, 51]
    remove_idx = [5, 15, 23]
    y_data_filtered = np.delete(y_data, remove_idx, axis=0)
    print("type(cnn_data)")
    print(type(cnn_data))
    print("cnn_data.shape")
    print(cnn_data.shape)

    cnn_data_filtered = np.delete(cnn_data, remove_idx, axis=0)
    lstm_data_filtered = np.delete(lstm_data, remove_idx, axis=0)
    transformer_data_filtered = np.delete(transformer_data, remove_idx, axis=0)

    np.save(ROOT_DIR + "/haptic_data/data3/normalized_data_filtered_2.npy", y_data_filtered)
    np.save(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1_filtered_2.npy", cnn_data_filtered)
    np.save(ROOT_DIR + "/recurrent/dataset3_results/data3_pred_lstm_v1_2_filtered_2.npy", lstm_data_filtered)
    np.save(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1_filtered_2.npy", transformer_data_filtered)

    # print("type(data)")
    # print(type(data))
    # print("data.shape")
    # print(data.shape)
