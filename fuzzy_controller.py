import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyGainController:
    def __init__(self):
        self.gain_sim = self._build_fuzzy_system()
        self.e_0_mean = None
        self.de_pre = 0.0

    def _build_fuzzy_system(self):
        error = ctrl.Antecedent(np.linspace(0, 1, 100), 'error')
        delta_error = ctrl.Antecedent(np.linspace(-1, 1, 100), 'delta_error')
        gain_fuzzy = ctrl.Consequent(np.linspace(0, 2, 100), 'gain_fuzzy', defuzzify_method='centroid')

        # Membership Functions
        self._set_membership(error, [0.0, 0.5, 1.0])
        self._set_membership(delta_error, [-1.0, 0.0, 1.0])

        gain_fuzzy['low'] = fuzz.gaussmf(gain_fuzzy.universe, 0.0, 0.15)
        gain_fuzzy['medium'] = fuzz.gaussmf(gain_fuzzy.universe, 1.0, 0.15)
        gain_fuzzy['high'] = fuzz.gaussmf(gain_fuzzy.universe, 2.0, 0.15)

        rules = [
            ctrl.Rule(error['low'] & delta_error['low'], gain_fuzzy['high']),
            ctrl.Rule(error['low'] & delta_error['medium'], gain_fuzzy['high']),
            ctrl.Rule(error['low'] & delta_error['high'], gain_fuzzy['high']),
            ctrl.Rule(error['medium'] & delta_error['low'], gain_fuzzy['low']),
            ctrl.Rule(error['medium'] & delta_error['medium'], gain_fuzzy['medium']),
            ctrl.Rule(error['medium'] & delta_error['high'], gain_fuzzy['low']),
            ctrl.Rule(error['high'], gain_fuzzy['low']), 
        ]
        return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def _set_membership(self, variable, centers):
        variable['low'] = fuzz.gbellmf(variable.universe, a=0.3, b=2, c=centers[0])
        variable['medium'] = fuzz.gbellmf(variable.universe, a=0.3, b=2, c=centers[1])
        variable['high'] = fuzz.gbellmf(variable.universe, a=0.3, b=2, c=centers[2])

    def compute_gain(self, e_now, e_prev, dt):
        if self.e_0_mean is None:
            self.e_0_mean = e_now
            self.de_pre = 0.0

            # 防止出現除以零的情況
            if self.e_0_mean == 0: self.e_0_mean = 1.0 

        if e_prev == 0.0 or dt == 0:
            de = e_now
        else:
            de = np.abs(e_now - e_prev) / dt
        
        self.de_pre = de

        # 標準化
        e_norm = e_now / self.e_0_mean
        de_norm = de / self.e_0_mean

        self.gain_sim.input['error'] = e_norm
        self.gain_sim.input['delta_error'] = de_norm
        self.gain_sim.compute()
        
        return self.gain_sim.output['gain_fuzzy']