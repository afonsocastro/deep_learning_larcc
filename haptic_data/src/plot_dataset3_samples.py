import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
from config.definitions import ROOT_DIR

class DataPlotter:
    def __init__(self, data, measurements):
        self.data = data
        self.measurements = measurements
        self.idx = 0

        self.fig, self.ax = plt.subplots(3, 1)
        self.lines = []

        self.create_buttons()
        self.update_graph()

        plt.show()

    def create_buttons(self):
        axnext = plt.axes([0.8, 0.025, 0.1, 0.05])
        axprev = plt.axes([0.1, 0.025, 0.1, 0.05])

        self.bnext = Button(axnext, '+1')
        self.bprev = Button(axprev, '-1')

        self.bnext.on_clicked(self.next1)
        self.bprev.on_clicked(self.prev1)

    def update_graph(self):
        for axis in self.ax:
            axis.cla()
            axis.grid()


        # labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
        # label_idx = int(self.data[self.idx, -1])
        # label_name = labels[label_idx]

        x_axis = np.linspace(0, 15, self.measurements)

        data_array = self.data[self.idx, : ,  :-1]

        graph_color = ["-r", "-g", "-b", "-y", "-k", "-m",
                       "-r", "-g", "-b", "-r", "-g", "-b"]

        # --- Joints ---
        joints_max = []
        for i in range(0, 6):
            joints_max.append(max(abs(data_array[:, i])))
            self.ax[0].plot(x_axis, data_array[:, i], graph_color[i])

        self.ax[0].set_title("Joints efforts")
        self.ax[0].legend(["J0", "J1", "J2", "J3", "J4", "J5"])
        self.ax[0].set_ylim((-1.1*max(joints_max), 1.1*max(joints_max)))

        # --- Forces ---
        forces_max = []
        for i in range(6, 9):
            forces_max.append(max(abs(data_array[:, i])))
            self.ax[1].plot(x_axis, data_array[:, i], graph_color[i])

        self.ax[1].set_title("Gripper Forces")
        self.ax[1].legend(["Fx", "Fy", "Fz"])
        self.ax[1].set_ylim((-1.1*max(forces_max), 1.1*max(forces_max)))

        # --- Torques ---
        torques_max = []
        for i in range(9, 12):
            torques_max.append(max(abs(data_array[:, i])))
            self.ax[2].plot(x_axis, data_array[:, i], graph_color[i])

        self.ax[2].set_title("Gripper Moments")
        self.ax[2].legend(["Mx", "My", "Mz"])
        self.ax[2].set_ylim((-1.1*max(torques_max), 1.1*max(torques_max)))
        self.ax[2].set_xlabel("Time (s)")


        self.fig.suptitle(f"Sample {self.idx}", fontsize=16)

        self.fig.canvas.draw_idle()

    # --- Callbacks ---
    def next1(self, event):
        self.idx = min(self.idx + 1, self.data.shape[0] - 1)
        self.update_graph()

    def prev1(self, event):
        self.idx = max(self.idx - 1, 0)
        self.update_graph()

if __name__ == "__main__":
    data = np.load(ROOT_DIR + "/haptic_data/data3/normalized_data_15s.npy")
    DataPlotter(data, measurements=1500)