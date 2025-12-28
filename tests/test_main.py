import unittest
import numpy as np
import sys
import os
import time
from unittest.mock import MagicMock, patch, call

# 設定路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================= 1. Mock 外部硬體依賴 =================
sys.modules['RTDE_func'] = MagicMock()
sys.modules['Rec_Cmd_func'] = MagicMock()
sys.modules['rs_func'] = MagicMock()
sys.modules['pupil_apriltags'] = MagicMock()
sys.modules['keyboard'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# ================= 2. Import main =================
import main
from config import Config

class TestMainFlow(unittest.TestCase):
    
    def setUp(self):
        pass

    @patch('main.RTDE_func')
    @patch('main.Rec_Cmd_func')
    @patch('main.get_camera')
    @patch('main.apriltag.Detector')
    @patch('main.DataLogger')
    @patch('main.AsyncRecorder')
    @patch('main.VisualServoController')
    @patch('main.TrajectoryPlot')
    def test_main_execution_flow(self, MockPlotter, MockController, MockRecorder, MockLogger, MockDetector, MockGetCamera, MockRecCmd, MockRTDE):
        """
        測試主程式的完整流程
        """
        
        # --- A. 設置 Mock 物件的行為 ---
        
        # 1. 模擬 RTDE
        mock_rtde_con = MagicMock()
        MockRTDE.rtde_init.return_value = mock_rtde_con
        mock_state = MagicMock()
        mock_state.actual_q = np.zeros(6)
        mock_state.actual_TCP_pose = np.zeros(6)
        mock_rtde_con.receive.return_value = mock_state
        
        # 2. [關鍵修正] 模擬相機 (Realsense) 及其參數
        mock_camera = MagicMock()
        MockGetCamera.return_value = mock_camera
        mock_camera.get_RGB_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # [新增] 設定相機內參為具體的浮點數，避免 TypeError
        mock_camera.rs_para.RGB_Cu = 320.0
        mock_camera.rs_para.RGB_Cv = 240.0
        mock_camera.rs_para.RGB_fu = 600.0
        mock_camera.rs_para.RGB_fv = 600.0
        
        # 3. 模擬 AprilTag 偵測
        mock_detector_instance = MockDetector.return_value
        mock_tag = MagicMock()
        mock_tag.corners = np.array([[0,0], [10,0], [10,10], [0,10]])
        mock_detector_instance.detect.return_value = [mock_tag]
        
        # 4. 模擬 Controller
        mock_controller_instance = MockController.return_value
        mock_controller_instance.update.side_effect = [
            (np.zeros(6), [10.0]*4, 0.1, 0),
            (np.zeros(6), [1.0]*4, 0.1, 0), 
        ]

        # --- B. 執行 main() ---
        try:
            main.main()
        except StopIteration:
            pass

        # --- C. 驗證行為 ---
        MockRTDE.rtde_init.assert_called()
        MockRTDE.servo_on.assert_called()
        MockRecorder.return_value.start.assert_called()
        self.assertTrue(MockRTDE.set_command["V_xyz"].called)
        
        # 驗證存檔與畫圖是否被呼叫
        # (因為 base_dir 已經修復，且相機參數正確，這裡應該會成功執行)
        MockLogger.return_value.save_all_to_csv.assert_called_once()
        MockPlotter.return_value.plot_all.assert_called_once()
        
        MockRTDE.servo_off.assert_called()

    @patch('main.RTDE_func')
    @patch('main.get_camera')
    @patch('main.apriltag.Detector')
    @patch('main.AsyncRecorder')
    @patch('main.VisualServoController')
    def test_tag_loss_handling(self, MockController, MockRecorder, MockDetector, MockGetCamera, MockRTDE):
        """測試 Tag 遺失處理邏輯"""
        original_limit = Config.NO_TAG_LIMIT
        Config.NO_TAG_LIMIT = 3
        
        try:
            # 模擬偵測失敗 (空 list)
            MockDetector.return_value.detect.return_value = []
            
            # [修正] 同樣要設定相機參數，雖然這裡可能跑不到 uv_to_normalized_xy 的數學計算部分
            mock_camera = MagicMock()
            MockGetCamera.return_value = mock_camera
            mock_camera.get_RGB_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_camera.rs_para.RGB_Cu = 320.0
            mock_camera.rs_para.RGB_Cv = 240.0
            mock_camera.rs_para.RGB_fu = 600.0
            mock_camera.rs_para.RGB_fv = 600.0
            
            main.main()
            
            # 沒偵測到 Tag，Controller 不應被更新
            MockController.return_value.update.assert_not_called()
            MockRTDE.servo_off.assert_called()
            
        finally:
            Config.NO_TAG_LIMIT = original_limit

if __name__ == '__main__':
    unittest.main()