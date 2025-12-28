import numpy as np
import threading
import time

class DataLogger:
    def __init__(self, paths):
        self.paths = paths
        self.image_move_data = []
        self.velocity_cmd_data = []
        self.distance_err_data = []
        self.gains_data = []
        self.memristor_data = []
        
        # Thread data
        self.thread_vel_cmd = []
        self.thread_dist_err = []
        self.thread_time = []
        self.thread_q = []
        self.thread_dq = []
        self.thread_pose = []

    def append_control_data(self, image_xy, vel_cmd, error_list, gain, memristor_val=0):
        self.image_move_data.append(image_xy)
        self.velocity_cmd_data.append(vel_cmd)
        self.distance_err_data.append(error_list)
        self.gains_data.append(gain)
        self.memristor_data.append(memristor_val)

    def save_all_to_csv(self):
        print("Saving data to CSV")
        try:
            np.savetxt(self.paths['image_moving'], self.image_move_data, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['velocity_cmd'], self.velocity_cmd_data, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['distance_err'], self.distance_err_data, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['gain'], self.gains_data, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['memristor'], self.memristor_data, delimiter=",", fmt="%.4f")
            
            # Thread data
            np.savetxt(self.paths['vel_cmd_thread'], self.thread_vel_cmd, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['dist_err_thread'], self.thread_dist_err, delimiter=",", fmt="%.4f")
            np.savetxt(self.paths['time_thread'], self.thread_time, delimiter=",", fmt="%.4f")
            print("Data saving complete.")
        except Exception as e:
            print(f"Error saving data: {e}")

class AsyncRecorder:
    def __init__(self, rtde, logger, interval=0.01):
        self.rtde = rtde
        self.logger = logger
        self.interval = interval
        self.running = False
        self.thread = threading.Thread(target=self._recording_loop, daemon=True)
        
        self.current_error = [0,0,0,0]
        self.current_vc = np.zeros(6)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def update_state(self, error, vc):
        self.current_error = error
        self.current_vc = vc

    def _recording_loop(self):
        start_time = time.time()
        while self.running:
            
            try:
                state = self.rtde.receive()
                self.logger.thread_dist_err.append(self.current_error)
                self.logger.thread_vel_cmd.append(self.current_vc)
                self.logger.thread_time.append(time.time() - start_time)

                if state.runtime_state == 2:
                    self.logger.thread_q.append(state.actual_q)
                    self.logger.thread_dq.append(state.actual_dq)
                    self.logger.thread_pose.append(state.actual_TCP_pose)
            except Exception as e:
                print(f"Recorder Error: {e}")

            time.sleep(self.interval)