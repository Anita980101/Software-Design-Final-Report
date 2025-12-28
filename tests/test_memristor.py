import unittest
import sys
import os  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memristor import Memristor

class TestMemristor(unittest.TestCase):
    
    def test_basic_output(self):
        """測試基本的累積與輸出計算"""
        # m_value=0.1, saturation=100
        mem = Memristor(m_value=0.1, saturation=100.0)
        
        # Step 1: Input 10.0 -> mtmp 變成 10.0
        # Output = 0.1 * abs(10.0) = 1.0
        out1 = mem.get_output(10.0)
        self.assertEqual(out1, 1.0)
        self.assertEqual(mem.mtmp, 10.0)
        
        # Step 2: Input -5.0 -> mtmp 變成 5.0
        # Output = 0.1 * abs(5.0) = 0.5
        out2 = mem.get_output(-5.0)
        self.assertEqual(out2, 0.5)
        self.assertEqual(mem.mtmp, 5.0)

    def test_saturation_upper(self):
        """測試上限飽和"""
        mem = Memristor(m_value=1.0, saturation=10.0)
        
        # 輸入 20，應該被限制在 10
        mem.get_output(20.0)
        self.assertEqual(mem.mtmp, 10.0)
        
        # 再輸入 5，應該還是 10 (因為 10+5 > 10)
        mem.get_output(5.0)
        self.assertEqual(mem.mtmp, 10.0)

    def test_saturation_lower(self):
        """測試下限飽和"""
        mem = Memristor(m_value=1.0, saturation=10.0)
        
        # 輸入 -20，應該被限制在 -10
        mem.get_output(-20.0)
        self.assertEqual(mem.mtmp, -10.0)

    def test_reset(self):
        """測試重置功能"""
        mem = Memristor()
        mem.get_output(10.0)
        
        self.assertNotEqual(mem.mtmp, 0.0)
        self.assertTrue(len(mem.total_output) > 0)
        
        mem.reset()
        
        self.assertEqual(mem.mtmp, 0.0)
        self.assertEqual(len(mem.total_output), 0)
        self.assertEqual(len(mem.total_ristor), 0)

    def test_history_recording(self):
        """測試歷史數據是否有被記錄"""
        mem = Memristor()
        mem.get_output(1.0)
        mem.get_output(2.0)
        
        self.assertEqual(len(mem.get_all_output()), 2)
        self.assertEqual(len(mem.get_total_ristor()), 2)
        # 第一次 mtmp=1.0, 第二次 mtmp=3.0
        self.assertEqual(mem.get_total_ristor(), [1.0, 3.0])

if __name__ == '__main__':
    unittest.main()