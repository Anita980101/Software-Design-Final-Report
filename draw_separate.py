import numpy as np
import matplotlib.pyplot as plt

save_name = "fuzzy"
OUTPUT_FILE_NAME = f'./data/data_1001/{save_name}'

class TrajectoryPlot:
    def __init__(self, image_trajectory_file, velocity_file, distance_error_file):
        self.image_trajectory_file = image_trajectory_file
        self.velocity_file = velocity_file
        self.distance_error_file = distance_error_file
        self.data = None
        self.data_velocity = None
        self.distance_error_data = None
        self.start_point = None
        self.end_point = None

    def load_data(self):
        self.data = np.loadtxt(self.image_trajectory_file, delimiter=',', skiprows=1)
        len_data = self.data.shape[0]
        self.start_point = [
            (self.data[1, 0], self.data[1, 1]),
            (self.data[1, 2], self.data[1, 3]),
            (self.data[1, 4], self.data[1, 5]),
            (self.data[1, 6], self.data[1, 7])
        ]
        self.end_point = [
            (self.data[len_data-1, 0], self.data[len_data-1, 1]),
            (self.data[len_data-1, 2], self.data[len_data-1, 3]),
            (self.data[len_data-1, 4], self.data[len_data-1, 5]),
            (self.data[len_data-1, 6], self.data[len_data-1, 7])
        ]

    def _remove_leading_zeros(self, data):
        index = 0
        while index < len(data) and np.all(data[index] == 0):
            index += 1
        return data[index:]
    def plot_all(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        # =================== 第一張：Trajectory ===================
        plt.figure()

        for i, color, label in zip(range(4), ['red', 'blue', 'orange', 'green'], ['point0', 'point1', 'point2', 'point3']):
            x_coords = self.data[:, 2 * i]
            y_coords = self.data[:, 2 * i + 1]
            plt.plot(x_coords, y_coords, color=color, marker='o', label=label, markersize=2)

        self._connect_points(self.start_point, plt, linestyle='--', color='grey')
        self._connect_points(self.end_point, plt, linestyle='-', color='black')

         #  畫出 desired goal 紅色框
        desired_goal = [
            (-0.1543, -0.173),
            (-0.166,  0.167),
            ( 0.174,  0.179),
            ( 0.1799, -0.1736)
        ]
        self._connect_points(desired_goal, plt, linestyle='-', color='purple', linewidth=1, label='desired goal')
        

        plt.title('Trajectory Plot from Start Point to End Point', fontsize=10)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.axis('equal')
        plt.savefig(f'{OUTPUT_FILE_NAME}/figure_1.png')
        plt.show()

        # =================== 第二張：Distance error ===================
        plt.figure()

        self.distance_error_data = np.loadtxt(self.distance_error_file, delimiter=',', skiprows=1)
        self.distance_error_data = self._remove_leading_zeros(self.distance_error_data)
        len_distance_error_data = self.distance_error_data.shape[0]
        x = np.arange(1, len_distance_error_data + 1)
        colors = ['red', 'blue', 'orange', 'green']
        labels = ['p0', 'p1', 'p2', 'p3']
        for i in range(4):
            e_temp = self.distance_error_data[:, i]
            plt.plot(x, e_temp, marker='o', linestyle='-', color=colors[i], label=labels[i], markersize=2, linewidth=0.5)

        plt.xlabel('Time')
        plt.ylabel('error(pixel)')
        plt.title('Distance error')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'{OUTPUT_FILE_NAME}/figure_2.png')
        plt.show()

        # =================== 第三張：Linear velocity ===================
        plt.figure()

        self.data_velocity = np.loadtxt(self.velocity_file, delimiter=',', skiprows=1)
        self.data_velocity = self._remove_leading_zeros(self.data_velocity)
        len_data_velocity = self.data_velocity.shape[0]
        x = np.arange(1, len_data_velocity + 1)
        colors = ['red', 'blue', 'orange']
        labels = ['vx', 'vy', 'vz']
        for i in range(3):
            v_temp = self.data_velocity[:, i]
            plt.plot(x, v_temp, marker='o', linestyle='-', color=colors[i], label=labels[i], markersize=2, linewidth=0.5)

        plt.xlabel('Time')
        plt.ylabel('Velocity (m/s)')
        plt.title('Linear Velocity Command')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'{OUTPUT_FILE_NAME}/figure_3.png')
        plt.show()

        # =================== 第四張：Angular velocity ===================
        plt.figure()

        colors = ['yellow', 'green', 'purple']
        labels = ['wx', 'wy', 'wz']
        for i in range(3, 6):
            v_temp = self.data_velocity[:, i]
            plt.plot(x, v_temp, marker='o', linestyle='-', color=colors[i - 3], label=labels[i - 3], markersize=2, linewidth=0.5)

        plt.xlabel('Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity Command')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'{OUTPUT_FILE_NAME}/figure_4.png')
        plt.show()


    def _connect_points(self, points, ax, **kwargs):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        for i in range(len(points)):
            # 只在第一條線段使用完整的 kwargs(包含 label)
            if i == 0:
                ax.plot([x_coords[i], x_coords[(i + 1) % len(points)]],
                        [y_coords[i], y_coords[(i + 1) % len(points)]], **kwargs)
            else:
                # 其他線段移除 label
                line_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
                ax.plot([x_coords[i], x_coords[(i + 1) % len(points)]],
                        [y_coords[i], y_coords[(i + 1) % len(points)]], **line_kwargs)
    # def _connect_points(self, points, ax, **kwargs):
    #     x_coords = [p[0] for p in points]
    #     y_coords = [p[1] for p in points]
    #     for i in range(len(points)):
    #         # 移除 label 避免畫多次圖例
    #         line_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
    #         ax.plot([x_coords[i], x_coords[(i + 1) % len(points)]],
    #                 [y_coords[i], y_coords[(i + 1) % len(points)]], **line_kwargs)


if __name__ == "__main__":
    time="211005"
    num="8.218"
    distance_err_path =f'./data/data_1001/實驗三/PI control/{save_name}/a=1/{num}/1141001_{time}_distance_error_thread.csv'
    image_moving_path = f'./data/data_1001/實驗三/PI control/{save_name}/a=1/{num}/1141001_{time}_image_moving.csv'
    velocity_command_path = f'./data/data_1001/實驗三/PI control/{save_name}/a=1/{num}/1141001_{time}_velocity_command_thread.csv'

    plotter = TrajectoryPlot(image_moving_path, velocity_command_path, distance_err_path)
    plotter.load_data()
    plotter.plot_all()
    print("========== Revised complete! ==========")