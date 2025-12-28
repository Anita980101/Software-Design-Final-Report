import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

class FuzzyGainController:
    def __init__(self, save_path='./data/test_data/gain_results.csv'):
        self.save_path = save_path

        # 記錄歷史資料
        self.gain_output = []
        self.e_norm_list = []
        self.de_norm_list = []

        # 建立 fuzzy 控制系統
        self._setup_fuzzy()

        # 初始誤差參考
        self.e_0_mean = None
        self.delta_e_max = None
        self.de_pre = 0.0

    def _setup_fuzzy(self):
        # 定義模糊變數
        self.error = ctrl.Antecedent(np.linspace(0, 1, 100), 'error')
        self.delta_error = ctrl.Antecedent(np.linspace(-1, 1, 100), 'delta_error')
        self.gain_fuzzy = ctrl.Consequent(np.linspace(0, 2, 100), 'gain_fuzzy', defuzzify_method='centroid')

        # 設定 membership function
        self.error['low'] = fuzz.gbellmf(self.error.universe, a=0.3, b=2, c=0.0)
        self.error['medium'] = fuzz.gbellmf(self.error.universe, a=0.3, b=2, c=0.5)
        self.error['high'] = fuzz.gbellmf(self.error.universe, a=0.3, b=2, c=1.0)

        self.delta_error['low'] = fuzz.gbellmf(self.delta_error.universe, a=0.3, b=2, c=-1.0)
        self.delta_error['medium'] = fuzz.gbellmf(self.delta_error.universe, a=0.3, b=2, c=0.0)
        self.delta_error['high'] = fuzz.gbellmf(self.delta_error.universe, a=0.3, b=2, c=1.0)

        self.gain_fuzzy['low'] = fuzz.gaussmf(self.gain_fuzzy.universe, 0.0, 0.15)
        self.gain_fuzzy['medium'] = fuzz.gaussmf(self.gain_fuzzy.universe, 1.0, 0.15)
        self.gain_fuzzy['high'] = fuzz.gaussmf(self.gain_fuzzy.universe, 2.0, 0.15)

        # rules = [
        #     ctrl.Rule(self.error['low'] & self.delta_error['low'], self.gain_fuzzy['high']),
        #     ctrl.Rule(self.error['low'] & self.delta_error['medium'], self.gain_fuzzy['high']),
        #     ctrl.Rule(self.error['low'] & self.delta_error['high'], self.gain_fuzzy['medium']),
        #     ctrl.Rule(self.error['medium'] & self.delta_error['low'], self.gain_fuzzy['high']),
        #     ctrl.Rule(self.error['medium'] & self.delta_error['medium'], self.gain_fuzzy['medium']),
        #     ctrl.Rule(self.error['medium'] & self.delta_error['high'], self.gain_fuzzy['low']),
        #     ctrl.Rule(self.error['high'] & self.delta_error['low'], self.gain_fuzzy['medium']),
        #     ctrl.Rule(self.error['high'] & self.delta_error['medium'], self.gain_fuzzy['low']),
        #     ctrl.Rule(self.error['high'] & self.delta_error['high'], self.gain_fuzzy['low']),
        # ]
        
        rules = [
            ctrl.Rule(self.error['low'] & self.delta_error['low'], self.gain_fuzzy['high']),
            ctrl.Rule(self.error['low'] & self.delta_error['medium'], self.gain_fuzzy['high']),
            ctrl.Rule(self.error['low'] & self.delta_error['high'], self.gain_fuzzy['high']),
            
            ctrl.Rule(self.error['medium'] & self.delta_error['low'], self.gain_fuzzy['low']),
            ctrl.Rule(self.error['medium'] & self.delta_error['medium'], self.gain_fuzzy['medium']),
            ctrl.Rule(self.error['medium'] & self.delta_error['high'], self.gain_fuzzy['low']),
            
            ctrl.Rule(self.error['high'] & self.delta_error['low'], self.gain_fuzzy['low']),
            ctrl.Rule(self.error['high'] & self.delta_error['medium'], self.gain_fuzzy['low']),
            ctrl.Rule(self.error['high'] & self.delta_error['high'], self.gain_fuzzy['low']),
        ]

        gain_ctrl = ctrl.ControlSystem(rules)
        self.gain_sim = ctrl.ControlSystemSimulation(gain_ctrl)

    def reset_reference(self, first_error_value):
        self.e_0_mean = first_error_value
        # self.delta_e_max = self.e_0_mean
        self.de_pre = 0.0

    def compute_gain(self, e_now, e_prev, dt):
        # e_now = np.mean(current_error_value)
        # e_total存dt分子
        if self.e_0_mean is None:
            self.reset_reference(e_now)

        if e_prev == 0.0 :
            de = e_now
            self.de_pre = de
        else:
            # e_prev = np.mean(previous_error_value)
            # e_minus = np.abs(e_now - e_prev)
            # if e_minus == 0.0:
            #     de = self.de_pre
            # else:
            #     de = e_minus / dt
            #     self.de_pre = de
            e_minus = np.abs(e_now - e_prev)
            de = e_minus / dt
            self.de_pre = de
        

        # 正規化
        e_norm = e_now / self.e_0_mean
        de_norm = de / self.e_0_mean

        # 模糊推論
        self.gain_sim.input['error'] = e_norm
        self.gain_sim.input['delta_error'] = de_norm
        self.gain_sim.compute()
        gain_value = self.gain_sim.output['gain_fuzzy']

        # 記錄
        self.gain_output.append(gain_value)
        self.e_norm_list.append(e_norm)
        self.de_norm_list.append(de_norm)

        return gain_value

    # def save_results(self):
    #     df_save = pd.DataFrame({
    #         'gain': self.gain_output,
    #         'e_norm': self.e_norm_list,
    #         'de_norm': self.de_norm_list
    #     })
    #     df_save.to_csv(self.save_path, index=False)
    #     print(f"已經將結果儲存到 {self.save_path}")

    def plot_gain(self):
        plt.plot(self.gain_output)
        plt.title("Fuzzy Gain Output Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Gain")
        plt.grid(True)
        plt.show()
