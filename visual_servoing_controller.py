import numpy as np
import time
from config import Config

class VisualServoController:
    def __init__(self, fuzzy_ctrl=None, memristor_ctrl=None):
        self.cmd_pos = np.array(Config.COMMAND_POSITION)
        self.fuzzy_controller = fuzzy_ctrl
        self.memristor_controller = memristor_ctrl
        
        self.kp = Config.KP_DEFAULT
        self.ki = Config.KI_DEFAULT
        self.mode = Config.CONTROL_MODE
        self.pre_error = 0.0
        self.error_sum = np.zeros(8) 
        self.t_prev = time.time()
        self.start_time = None
        
        self.current_pixel_error_list = [0,0,0,0]
        self.c_v_c = np.zeros(6)
        self.phi = np.zeros(6)
        self.eta = np.zeros(6)
        self.is_first_run = True

    def calculate_interaction_matrix(self, features, Z):
        Le_rows = []
        for i in range(0, len(features), 2):
            u, v = features[i], features[i+1]
            row1 = [-1/Z, 0, u/Z, u*v, -(1+u**2), v]
            row2 = [0, -1/Z, v/Z, 1+v**2, -u*v, -u]
            Le_rows.append(row1)
            Le_rows.append(row2)
        return np.array(Le_rows)

    def update(self, image_features):
        t_now = time.time()
        if self.start_time is None: self.start_time = t_now
        dt = t_now - self.t_prev
        self.t_prev = t_now

        error_vector = self.cmd_pos - image_features
        self.current_pixel_error_list = []
        sum_square_error = 0
        pixel_error_sum_avg = 0.0
        
        # 計算 4 個特徵點的誤差
        for i in range(4):
            ex = error_vector[2*i]
            ey = error_vector[2*i+1]
            sq_err = ex**2 + ey**2
            sum_square_error += sq_err
            
            # focal length = 608
            p_err = np.sqrt(sq_err) * 608
            self.current_pixel_error_list.append(p_err)
            pixel_error_sum_avg += p_err
            
        e_total = np.sqrt(sum_square_error)
        avg_pixel_error = pixel_error_sum_avg / 4.0

        current_memristor_val = 0 

        # 根據模式選擇 Gain 的計算方式
        if self.mode == 'FUZZY':
            if self.fuzzy_controller:
                if self.pre_error == 0.0: self.pre_error = e_total
                self.kp = self.fuzzy_controller.compute_gain(e_total, self.pre_error, dt)
                self.pre_error = e_total
                
        elif self.mode == 'MEMRISTOR':
            if self.memristor_controller:
                self.kp = self.memristor_controller.get_output(avg_pixel_error)
                current_memristor_val = self.memristor_controller.total_ristor[-1]
                
        elif self.mode == 'PI':
            self.kp = Config.KP_DEFAULT
            self.error_sum += error_vector 
            
        elif self.mode == 'P': 
            self.kp = Config.KP_DEFAULT
            
        else:
            # 避免模式輸入錯誤
            raise ValueError(f"不存在的控制模式: {self.mode}")

        # 計算 Interaction Matrix
        Le = self.calculate_interaction_matrix(image_features, Config.Z_HEIGHT)
        pinv_Le = np.linalg.pinv(Le)

        if self.is_first_run:
            self.eta = pinv_Le @ error_vector
            self.is_first_run = False
            error_phi = 1.0
        else:
            error_phi = np.exp(-0.5 * (t_now - self.start_time))
            
        self.phi = self.kp * error_phi * self.eta

        # 控制律
        base_control = self.kp * (pinv_Le @ error_vector)
        
        if self.mode == 'PI':
            # PI 模式
            integral_term = self.ki * (pinv_Le @ self.error_sum)
            self.c_v_c = base_control + integral_term - self.phi
        else:
            # P, FUZZY, MEMRISTOR 模式
            self.c_v_c = base_control - self.phi
        
        return self.c_v_c, self.current_pixel_error_list, self.kp, current_memristor_val