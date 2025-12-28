import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Memristor_1 import Memresistor

class TestMemresistorLegacy(unittest.TestCase):
    
    def test_basic_output(self):
        """測試基本的累積與輸出計算 (Legacy)"""
        mem = Memresistor(m_value=0.1, saturation=100)
        
        # Step 1: Input 10
        # mtmp = 10, output = 0.1 * 10 = 1.0
        out1 = mem.get_output(10.0)
        self.assertEqual(out1, 1.0)
        self.assertEqual(mem.mtmp, 10.0)
        
        # Step 2: Input -5
        # mtmp = 5, output = 0.1 * 5 = 0.5
        out2 = mem.get_output(-5.0)
        self.assertEqual(out2, 0.5)

    def test_saturation(self):
        """測試飽和限制 (Legacy)"""
        mem = Memresistor(m_value=1.0, saturation=10)
        
        # 測試上限
        mem.get_output(20.0)
        self.assertEqual(mem.mtmp, 10.0)
        
        # 測試下限
        mem.get_output(-30.0) # 10 - 30 = -20 -> clamp to -10
        self.assertEqual(mem.mtmp, -10.0)

    def test_reset(self):
        """測試重置功能 (Legacy)"""
        mem = Memresistor()
        mem.get_output(10.0)
        self.assertTrue(len(mem.total_output) > 0)
        
        mem.reset()
        self.assertEqual(mem.mtmp, 0.0)
        self.assertEqual(len(mem.total_output), 0)

    def test_get_output_for(self):
        """測試舊版特有的迴圈模擬函式"""
        mem = Memresistor(m_value=0.01, saturation=100)
        # 輸入 0.01，跑 1000 次
        # 每次 mtmp 增加 0.01
        # 最後一次 mtmp 應約為 10.0
        # Output = 0.01 * 10.0 = 0.1
        final_output = mem.get_output_for(0.01)
        
        self.assertAlmostEqual(mem.mtmp, 10.0)
        self.assertAlmostEqual(final_output, 0.1)

if __name__ == '__main__':
    unittest.main()