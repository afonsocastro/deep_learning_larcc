#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from config.definitions import ROOT_DIR
from matplotlib.colors import ListedColormap
from utils import value_for_array, plot_shadow, grouping_segments


class StreamingDecisionEngine():
    def __init__(self, true_labels=None, plot=False, shake_threshold=0.3, entropy_threshold=0.5, cnn_weight=0.5,
                 transformer_weight=0.5, min_steady_timesteps=10):
        self.final_sequence = []

        self.is_counting = False
        self.new_primitive_counter = 0
        self.idx_where_counter_started = None

        self.new_primitive = None
        self.last_confirmed_primitive = None
        self.previous_primitive = None

        # Optimized parameters---------------------------0
        self.shake_threshold = shake_threshold
        self.entropy_threshold = entropy_threshold
        self.cnn_weight = cnn_weight
        self.transformer_weight = transformer_weight
        self.min_steady_timesteps = min_steady_timesteps
        # ---------------------------------------------

        self.draw_plot = False

        if plot:
            self.init_plot(true_labels)

    def predict_haptic_sequence(self, predictions, timestep):
        pull_cnn, push_cnn, shake_cnn, twist_cnn = predictions["cnn"][0], predictions["cnn"][1], predictions["cnn"][2], \
        predictions["cnn"][3]
        pull_trans, push_trans, shake_trans, twist_trans = predictions["transformer"][0], predictions["transformer"][1], \
        predictions["transformer"][2], predictions["transformer"][3]

        pull_sum = (self.cnn_weight * pull_cnn) + (self.transformer_weight * pull_trans)
        push_sum = (self.cnn_weight * push_cnn) + (self.transformer_weight * push_trans)
        shake_sum = (self.cnn_weight * shake_cnn) + (self.transformer_weight * shake_trans)
        twist_sum = (self.cnn_weight * twist_cnn) + (self.transformer_weight * twist_trans)

        if shake_sum > self.shake_threshold:
            pull_sum = 0.01
            push_sum = 0.01
            twist_sum = 0.01
            shake_sum = 0.97

        probs = np.array([pull_sum, push_sum, shake_sum, twist_sum])
        arr_safe = np.where(probs < 0.001, 0.001, probs)
        entropy = - np.sum(arr_safe * np.log2(arr_safe))

        if entropy > self.entropy_threshold:
            self.new_primitive = 4
        else:
            self.new_primitive = np.argmax(probs)

        if (self.new_primitive != self.previous_primitive) and (self.previous_primitive == self.last_confirmed_primitive):
           self.is_counting = True
           self.idx_where_counter_started = timestep

        elif (self.new_primitive != self.previous_primitive) and (
                self.previous_primitive != self.last_confirmed_primitive) and 0 < self.new_primitive_counter < self.min_steady_timesteps:
            if self.new_primitive != self.last_confirmed_primitive:
                self.last_confirmed_primitive = 4
                self.overwrite_recent(new_value=4, start=self.idx_where_counter_started, end=timestep)
            elif self.new_primitive == self.last_confirmed_primitive:
                self.last_confirmed_primitive = self.new_primitive
                self.overwrite_recent(new_value=self.new_primitive, start=self.idx_where_counter_started, end=timestep)
            self.is_counting = True
            self.idx_where_counter_started = timestep
            self.new_primitive_counter = 0

        if self.is_counting:
            self.new_primitive_counter += 1

            if self.new_primitive_counter == self.min_steady_timesteps:
                self.is_counting = False
                self.new_primitive_counter = 0
                self.last_confirmed_primitive = self.new_primitive

        if self.draw_plot:
            self.update_plot(timestep, probs, entropy)

        self.previous_primitive = self.new_primitive
        self.final_sequence.append(self.new_primitive)

    def init_plot(self, true_labels):
        self.window_size = 20
        self.draw_plot = True
        self.colors = {"pull": "blue", "push": "red", "shake": "green", "twist": "orange"}
        self.t_values = []
        self.pull_vals = []
        self.push_vals = []
        self.shake_vals = []
        self.twist_vals = []
        self.entropy_vals = []
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Streaming Haptic Prediction", fontsize=16)
        self.gs = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 1])
        self.axes = [self.fig.add_subplot(self.gs[0])]
        self.axes.append(self.fig.add_subplot(self.gs[1], sharex=self.axes[0]))
        self.axes.append(self.fig.add_subplot(self.gs[2], sharex=self.axes[0]))
        self.axes.append(self.fig.add_subplot(self.gs[3], sharex=self.axes[0]))
        self.start_line_1 = self.axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
        self.end_line_1 = self.axes[1].axvline(self.window_size, color='black', linestyle='--', linewidth=1.5)
        self.start_line_2 = self.axes[2].axvline(0, color='black', linestyle='--', linewidth=1.5)
        self.end_line_2 = self.axes[2].axvline(self.window_size, color='black', linestyle='--', linewidth=1.5)
        self.max_timesteps = len(true_labels)
        for ax in self.axes:
            ax.set_xlim(0, self.max_timesteps)

        plot_shadow(np.array([i for i in range(0, self.max_timesteps)]), true_labels, self.axes[0])
        self.axes[0].set_ylabel("Ground\nTruth")
        self.axes[0].set_yticklabels([])
        self.line_pull, = self.axes[1].plot([], [], color=self.colors["pull"], linewidth=2, label="pull")
        self.line_push, = self.axes[1].plot([], [], color=self.colors["push"], linewidth=2, label="push")
        self.line_shake, = self.axes[1].plot([], [], color=self.colors["shake"], linewidth=2, label="shake")
        self.line_twist, = self.axes[1].plot([], [], color=self.colors["twist"], linewidth=2, label="twist")
        self.axes[1].set_ylabel("Fused Probabilities")
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_ylim(-0.02, 1.02)
        self.line_entropy, = self.axes[2].plot([], [], color="purple", linewidth=2)
        self.axes[2].set_ylabel("Entropy")
        self.axes[2].set_ylim(-0.02, 2)

        self.pred_array = np.full(self.max_timesteps, np.nan)
        self.cmap = ListedColormap(["#ababf7", "#f4a6a6", "#a5e6a5", "#fdd9a0", "#c2c2c2"])
        self.pred_img = self.axes[3].imshow(self.pred_array[np.newaxis, :], aspect="auto", cmap=self.cmap, vmin=0,
                                            vmax=4, interpolation="nearest")
        self.pred_img.set_extent([0, self.max_timesteps, 0, 1])
        self.axes[3].set_ylabel("Predicted\nSequence")
        self.axes[3].set_yticks([])

        plt.ion()
        plt.show()

    def update_plot(self, timestep, probs, entropy):
        self.t_values.append(timestep + self.window_size - 1)
        self.pull_vals.append(probs[0])
        self.push_vals.append(probs[1])
        self.shake_vals.append(probs[2])
        self.twist_vals.append(probs[3])
        self.entropy_vals.append(entropy)
        self.line_pull.set_data(self.t_values, self.pull_vals)
        self.line_push.set_data(self.t_values, self.push_vals)
        self.line_shake.set_data(self.t_values, self.shake_vals)
        self.line_twist.set_data(self.t_values, self.twist_vals)
        self.line_entropy.set_data(self.t_values, self.entropy_vals)

        # --- ajustar eixo X dinamicamente ---
        # xmin, xmax = 0, max(self.t_values) + 1
        # self.axes[1].set_xlim(xmin, xmax)
        # self.axes[2].set_xlim(xmin, xmax)

        start = timestep
        end = timestep + self.window_size
        self.start_line_1.set_xdata(start)
        self.end_line_1.set_xdata(end)
        self.start_line_2.set_xdata(start)
        self.end_line_2.set_xdata(end)

        idx = timestep + self.window_size - 1
        if idx < self.max_timesteps:
            self.pred_array[idx] = self.new_primitive
            self.pred_img.set_data(self.pred_array[np.newaxis, :])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.pause(0.1)

    def overwrite_recent(self, new_value, start, end):
        end = min(end, len(self.final_sequence))
        if start < end:
            self.final_sequence[start:end] = [new_value] * (end - start)

            if self.draw_plot:
                idx_start = start + self.window_size - 1
                idx_end = end + self.window_size - 1

                idx_start = max(0, idx_start)
                idx_end = min(self.max_timesteps, idx_end)

                if idx_start < idx_end:
                    self.pred_array[idx_start:idx_end] = new_value
                    self.pred_img.set_data(self.pred_array[np.newaxis, :])

if __name__ == '__main__':
    time_steps = 1500
    sliding_window = 20
    sample = 7
    predictions = {"cnn": load(ROOT_DIR + "/convolutional/dataset3_results/data3_pred_cnn_v1_1.npy"),
                   "transformer": load(ROOT_DIR + "/transformers/dataset3_results/data3_pred_transformer_v1_1.npy")}
    y_data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    true_labels = y_data[sample, :, -1]

    ground_truth_sequence = grouping_segments(true_labels, delay=0)
    print("ground_truth_sequence")
    print(ground_truth_sequence)
    iterator_times = np.array([i for i in range(0, time_steps - sliding_window + 1)])

    outptus={"cnn": None, "transformer":None}

    sde = StreamingDecisionEngine(true_labels=true_labels,plot=True)
    # sde = StreamingDecisionEngine()

    for t_idx, ts in enumerate(iterator_times):
        outptus["cnn"] = predictions["cnn"][sample][ts]
        outptus["transformer"] = predictions["transformer"][sample][ts]

        sde.predict_haptic_sequence(outptus, ts)

    plt.ioff()
    plt.show()

    predicted_segments = grouping_segments(sde.final_sequence, delay=sliding_window-1)

    print("\npredicted_sequence")
    print(predicted_segments)

