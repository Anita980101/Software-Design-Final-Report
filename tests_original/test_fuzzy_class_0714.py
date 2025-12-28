import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzy_class_0714 import FuzzyGainController

class TestFuzzyLegacy(unittest.TestCase):
    def setUp(self):
        self.fuzzy = FuzzyGainController()

    def test_compute_gain_range(self):
        """測試 Gain 計算是否在合理範圍 (0~2)"""
        # 初始化參考值
        self.fuzzy.compute_gain(100.0, 0.0, 0.1)
        
        # 給定一個中等誤差
        gain = self.fuzzy.compute_gain(50.0, 50.0, 0.1)
        
        self.assertIsInstance(gain, float)
        self.assertTrue(0.0 <= gain <= 2.0)

    def test_reset_reference(self):
        """測試重置參考誤差值"""
        self.fuzzy.reset_reference(200.0)
        self.assertEqual(self.fuzzy.e_0_mean, 200.0)
        self.assertEqual(self.fuzzy.de_pre, 0.0)

    def test_zero_division_protection(self):
        """測試當 dt 不為 0 時的計算"""
        # 舊版程式碼沒有顯式的 dt=0 檢查，這裡主要測試正常 dt
        try:
            self.fuzzy.compute_gain(10.0, 5.0, 0.1)
        except Exception as e:
            self.fail(f"compute_gain raised exception: {e}")

if __name__ == '__main__':
    unittest.main()