import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzy_controller import FuzzyGainController

class TestFuzzyGainController(unittest.TestCase):
    def setUp(self):
        self.controller = FuzzyGainController()

    def test_gain_range(self):
        """測試模糊輸出是否在 0~2 之間"""
        gain = self.controller.compute_gain(e_now=10.0, e_prev=10.0, dt=0.1)
        self.assertGreaterEqual(gain, 0.0)
        self.assertLessEqual(gain, 2.0)

    def test_zero_dt(self):
        """測試 dt=0 防止崩潰"""
        try:
            self.controller.compute_gain(10.0, 5.0, 0)
        except ZeroDivisionError:
            self.fail("Fuzzy controller crashed on dt=0")

    def test_initial_state(self):
        """測試初始狀態"""
        self.assertIsNone(self.controller.e_0_mean)
        self.controller.compute_gain(100.0, 0.0, 0.1)
        self.assertEqual(self.controller.e_0_mean, 100.0)

    def test_fuzzy_logic_rules_high_error(self):
        """測試規則：高誤差 (Error High) -> 低增益 (Gain Low)"""
        # 設定非常大的誤差，標準化後會落入 High 的歸屬函數
        # 注意：你的 code 會將 error 除以 e_0_mean 進行標準化
        e_now = 200.0
        e_prev = 190.0
        dt = 0.1
        
        # 先做一次初始化 e_0_mean
        self.controller.compute_gain(200.0, 0, 0.1) 
        
        # 第二次計算，這時 error/e_0_mean = 1.0 (High)
        gain = self.controller.compute_gain(e_now, e_prev, dt)
        
        # 根據規則，Gain 應該偏低 (接近 0.0)
        # 這裡我們寬鬆檢查，確保它不會是 High (2.0)
        self.assertLess(gain, 1.0)

    def test_fuzzy_logic_rules_low_error(self):
        """測試規則：低誤差 (Error Low) & 低變化率 -> 高增益 (Gain High)"""
        # 初始化 e_0_mean 為大數值
        self.controller.compute_gain(100.0, 0, 0.1)
        
        # 輸入很小的誤差 (e_norm << 1)
        gain = self.controller.compute_gain(1.0, 1.0, 0.1)
        
        # 根據規則，Gain 應該偏高 (接近 2.0)
        self.assertGreater(gain, 1.0)

if __name__ == '__main__':
    unittest.main()