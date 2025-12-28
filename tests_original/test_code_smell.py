import unittest
import numpy as np
import sys
import os
import time
from unittest.mock import MagicMock, patch

# ================= 重要：Mock 缺失的外部依賴 =================
# 在 import code_smell 之前，必須先欺騙 Python 這些模組存在
# 這樣才能順利 import code_smell 檔案
sys.modules['RTDE_func'] = MagicMock()
sys.modules['Rec_Cmd_func'] = MagicMock()
sys.modules['gen_Trajectory'] = MagicMock()
sys.modules['rs_func'] = MagicMock()
sys.modules['pupil_apriltags'] = MagicMock()
# ==========================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import code_smell

class TestCodeSmellLogic(unittest.TestCase):
    
    def setUp(self):
        # 1. 初始化 VisualServoing 實例
        self.vs = code_smell.VisualServoing()
        
        # 2. [修正 Error 1] 手動注入 rtde 到 code_smell 模組
        # 因為 code_smell 中很多函式直接使用全域變數 'rtde'
        code_smell.rtde = MagicMock()
        
        # 3. [修正 Error 2] 初始化時間變數
        # visual_servoing 函式會計算 dt = now - t_prev，若為 None 會崩潰
        self.vs.t_prev = time.time()
        
        # 4. 將全域變數 vs 替換為我們的測試實例
        code_smell.vs = self.vs

    def test_kinematics_Rij(self):
        """測試旋轉矩陣計算 (Rij)"""
        # 測試 theta=0, i=0 (alpha=pi/2)
        # Rij should be [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        R = self.vs.Rij(0, 0)
        self.assertEqual(R.shape, (3, 3))
        self.assertAlmostEqual(R[0, 0], 1.0)
        
    def test_kinematics_EndEffectorToBase(self):
        """測試正向運動學矩陣"""
        theta = np.zeros(6)
        R_EB = self.vs.EndEffectorToBase(theta)
        self.assertEqual(R_EB.shape, (3, 3))

    def test_visual_servoing_logic(self):
        """測試視覺伺服控制律 (Interaction Matrix)"""
        # 模擬輸入
        image_xy = np.zeros(8) 
        Z = 0.31
        c_v_c = code_smell.visual_servoing(image_xy, Z)
        self.assertEqual(c_v_c.shape, (6,))
        # 檢查是否更新了 vs 的屬性
        self.assertEqual(len(self.vs.command_cVc), 1)
        self.assertEqual(len(self.vs.distance_err), 1)

    def test_camera_to_base_velocity(self):
        """測試相機速度轉基座速度"""
        # 設定 Mock RTDE 的回傳值
        mock_state = MagicMock()
        mock_state.actual_q = np.zeros(6)
        code_smell.rtde.receive.return_value = mock_state
        
        c_v_c = np.zeros(6)
        # 這裡直接呼叫，因為 setUp 已經幫我們把 rtde 塞進去了
        b_v_c = self.vs.CameratoBase_v(c_v_c)
            
        self.assertEqual(b_v_c.shape, (6,))

    # [修正 Error 3] 完整 Mock cv2，因為我們要開啟 draw=True
    @patch('code_smell.cv2') 
    def test_uv_to_camera_xy(self):
        """測試影像座標轉換"""
        mock_camera = MagicMock()
        mock_camera.rs_para.RGB_Cu = 320
        mock_camera.rs_para.RGB_Cv = 240
        mock_camera.rs_para.RGB_fu = 600
        mock_camera.rs_para.RGB_fv = 600
        
        # Mock Tag Detector
        mock_detector = MagicMock()
        mock_tag = MagicMock()
        mock_tag.corners = np.array([[0,0], [10,0], [10,10], [0,10]])
        mock_tag.center = np.array([5,5])
        mock_detector.detect.return_value = [mock_tag]
        
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        flag, tags, features = code_smell.uv_to_camera_xy(img, mock_detector, mock_camera, draw=True)
        
        self.assertTrue(flag)
        self.assertEqual(len(features), 8) 
        self.assertAlmostEqual(features[0], (0-320)/600) 

if __name__ == '__main__':
    unittest.main()