import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config  # 引入 Config 確保目標一致

class TrajectoryPlot:
    def __init__(self, image_traj_path, velocity_path, dist_err_path, output_dir):
        self.image_traj_path = image_traj_path
        self.velocity_path = velocity_path
        self.dist_err_path = dist_err_path
        self.output_dir = output_dir
        
        self.data = None
        self.data_velocity = None
        self.distance_error_data = None
        self.start_point = None
        self.end_point = None

    def load_data(self):
        try:
            self.data = np.loadtxt(self.image_traj_path, delimiter=',', skiprows=1)
            # 處理只有一行數據的情況
            if self.data.ndim == 1:
                self.data = self.data.reshape(1, -1)
                
            len_data = self.data.shape[0]
            if len_data < 2:
                print("數據點太少")
                return

            self.start_point = self._get_points_from_row(1)
            self.end_point = self._get_points_from_row(len_data - 1)
            print("Data loaded successfully.")
            
        except Exception as e:
            print(f"Loading image data failed: {e}")

    def _get_points_from_row(self, row_idx):
        row = self.data[row_idx]
        return [
            (row[0], row[1]), (row[2], row[3]),
            (row[4], row[5]), (row[6], row[7])
        ]

    def _remove_leading_zeros(self, data):
        if data is None or len(data) == 0: return data
        idx = 0
        while idx < len(data) and np.all(data[idx] == 0):
            idx += 1
        return data[idx:]

    def plot_all(self):
        if self.data is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        self._plot_trajectory()
        self._plot_distance_error()
        self._plot_velocity()

    def _plot_trajectory(self):
        plt.figure()
        colors = ['red', 'blue', 'orange', 'green']
        labels = ['point0', 'point1', 'point2', 'point3']

        for i in range(4):
            x = self.data[:, 2 * i]
            y = self.data[:, 2 * i + 1]
            plt.plot(x, y, color=colors[i], marker='o', label=labels[i], markersize=2)

        if self.start_point:
            self._connect_points(self.start_point, plt, linestyle='--', color='grey')
        if self.end_point:
            self._connect_points(self.end_point, plt, linestyle='-', color='black')

        desired_goal = Config.get_goal_points_for_plot()
        self._connect_points(desired_goal, plt, linestyle='-', color='purple', linewidth=1, label='Desired Goal')

        plt.title('Image Trajectory', fontsize=10)
        plt.xlabel('u', fontsize=12)
        plt.ylabel('v', fontsize=12)
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.axis('equal')
        plt.savefig(f'{self.output_dir}/trajectory.png')

    def _plot_distance_error(self):
        try:
            raw_data = np.loadtxt(self.dist_err_path, delimiter=',', skiprows=1)
            data = self._remove_leading_zeros(raw_data)
            
            plt.figure()
            x = np.arange(1, len(data) + 1)
            colors = ['red', 'blue', 'orange', 'green']
            for i in range(4):
                if i < data.shape[1]:
                    plt.plot(x, data[:, i], color=colors[i], label=f'p{i}', linewidth=0.5)

            plt.xlabel('Time Step')
            plt.ylabel('Error (pixel)')
            plt.title('Distance Error')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/error.png')
        except Exception as e:
            print(f"Plotting Error failed: {e}")

    def _plot_velocity(self):
        try:
            raw_data = np.loadtxt(self.velocity_path, delimiter=',', skiprows=1)
            data = self._remove_leading_zeros(raw_data)
            if data.ndim == 1: data = data.reshape(1, -1)
            
            x = np.arange(1, len(data) + 1)
            
            # Linear Velocity
            plt.figure()
            for i, label in enumerate(['vx', 'vy', 'vz']):
                plt.plot(x, data[:, i], label=label, linewidth=0.5)
            plt.title('Linear Velocity Command')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/linear_velocity.png')

            # Angular Velocity
            plt.figure()
            for i, label in enumerate(['wx', 'wy', 'wz']):
                plt.plot(x, data[:, i+3], label=label, linewidth=0.5)
            plt.title('Angular Velocity Command')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/angular_velocity.png')
            
        except Exception as e:
            print(f"Plotting Velocity failed: {e}")

    def _connect_points(self, points, ax, **kwargs):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        for i in range(len(points)):
            args = kwargs if i == 0 else {k:v for k,v in kwargs.items() if k != 'label'}
            ax.plot([x[i], x[(i+1)%len(points)]], 
                    [y[i], y[(i+1)%len(points)]], **args)