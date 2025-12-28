import unittest
import numpy as np
import sys
import os

# 將上層目錄加入 path 以便 import 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics import RobotKinematics

class TestRobotKinematics(unittest.TestCase):
    
    def test_get_Rij_shape(self):
        """測試 DH 矩陣維度 (3x3)"""
        theta = np.pi / 2
        R = RobotKinematics.get_Rij(theta, 0)
        self.assertEqual(R.shape, (3, 3))
        # 驗證數值 (cos(90)=0, sin(90)=1)
        # 根據您的 DH 表，alph[0] = pi/2
        # R[0,0] = cos(theta) = 0
        self.assertAlmostEqual(R[0, 0], 0.0)

    def test_end_effector_to_base(self):
        """測試正向運動學矩陣維度"""
        q = np.zeros(6)
        R_EB = RobotKinematics.end_effector_to_base(q)
        self.assertEqual(R_EB.shape, (3, 3))

    def test_camera_velocity_to_base(self):
        """測試速度轉換矩陣維度 (輸入 6x1 -> 輸出 6x1)"""
        c_v_c = np.array([0.1, 0, 0, 0, 0, 0])
        current_q = np.zeros(6)
        
        vel_cmd = RobotKinematics.camera_velocity_to_base(c_v_c, current_q)
        
        self.assertTrue(vel_cmd.shape == (6,) or vel_cmd.shape == (6, 1))

    def test_Rij_identity(self):
        """測試：當 theta=0 且 alpha=0 時，應為單位矩陣 (Identity Matrix)"""
        R = RobotKinematics.get_Rij(0, 1)
        
        self.assertAlmostEqual(R[0, 0], 1.0) # cos(0)
        self.assertAlmostEqual(R[1, 1], 1.0) # cos(0)*cos(0)
        self.assertAlmostEqual(R[2, 2], 1.0) # cos(0)
        self.assertAlmostEqual(R[0, 1], 0.0) # -sin(0)
        self.assertAlmostEqual(R[1, 0], 0.0) # sin(0)

    def test_camera_to_end_effector_structure(self):
        """測試：手眼校正矩陣的結構"""
        matrix = RobotKinematics.camera_to_end_effector()
        
        # 檢查維度是否為 6x6
        self.assertEqual(matrix.shape, (6, 6))
        
        # 檢查左下角 3x3 區域是否全為 0 (物理上，線速度不應影響角速度)
        lower_left = matrix[3:6, 0:3]
        self.assertTrue(np.all(lower_left == 0))

if __name__ == '__main__':
    unittest.main()