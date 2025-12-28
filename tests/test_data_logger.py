import unittest
import numpy as np
import sys
import os
import time
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_logger import DataLogger, AsyncRecorder

class TestDataLogger(unittest.TestCase):
    def setUp(self):
        # 建立假的 paths 字典
        self.paths = {k: f'{k}.csv' for k in ['image_moving', 'velocity_cmd', 'distance_err', 'gain', 'memristor', 'vel_cmd_thread', 'dist_err_thread', 'time_thread']}
        self.logger = DataLogger(self.paths)

    def test_append_data(self):
        """測試數據 list 是否正確增加"""
        self.logger.append_control_data([0]*8, [0]*6, [0.1]*4, 0.5)
        self.assertEqual(len(self.logger.image_move_data), 1)
        self.assertEqual(self.logger.gains_data[0], 0.5)

    @patch('numpy.savetxt')
    def test_save_csv(self, mock_savetxt):
        """測試是否呼叫 numpy.savetxt"""
        self.logger.save_all_to_csv()
        self.assertTrue(mock_savetxt.called)

class TestAsyncRecorder(unittest.TestCase):
    def setUp(self):
        self.mock_rtde = MagicMock()
        self.mock_logger = MagicMock()
        # 初始化清單以便 append
        self.mock_logger.thread_dist_err = []
        self.mock_logger.thread_vel_cmd = []
        self.mock_logger.thread_time = []
        
        self.recorder = AsyncRecorder(self.mock_rtde, self.mock_logger, interval=0.001)

    def test_update_state(self):
        """測試狀態更新"""
        err = [1, 2, 3, 4]
        vc = np.ones(6)
        self.recorder.update_state(err, vc)
        self.assertEqual(self.recorder.current_error, err)
        np.testing.assert_array_equal(self.recorder.current_vc, vc)

    def test_recording_logic(self):
        """測試執行緒邏輯 (不真正啟動 Thread，直接測迴圈邏輯)"""
        # 模擬 RTDE 回傳
        mock_state = MagicMock()
        mock_state.runtime_state = 2
        mock_state.actual_q = [0]*6
        self.mock_rtde.receive.return_value = mock_state
        
        # 手動觸發一次迴圈內容
        self.recorder.running = True
        # 這裡我們不呼叫 _recording_loop 的 while True，而是模擬迴圈內的行為
        # 為了測試方便，我們直接呼叫 start/stop 並讓它跑一小段時間
        
        self.recorder.start()
        time.sleep(0.02) # 讓子執行緒跑一下
        self.recorder.stop()
        
        # 檢查是否有數據被寫入 logger
        self.assertGreater(len(self.mock_logger.thread_time), 0)

    @patch('numpy.savetxt')
    def test_save_empty_data(self, mock_savetxt):
        """測試：當沒有任何數據時，呼叫存檔是否會崩潰"""
        # 不呼叫 append，直接 save
        try:
            self.logger.save_all_to_csv()
        except Exception as e:
            self.fail(f"Saving empty data raised exception: {e}")
            
        # 即使是空的，savetxt 仍可能被呼叫 (視實作而定)，
        # 重點是不能 Crash

    def test_append_mismatched_dimensions(self):
        """測試：(防禦性程式設計) 如果傳入的維度不對，Logger 是否照單全收"""
        # 你的 Logger 目前是直接 append，這個測試是用來提醒未來的驗證需求
        # 或者確保即使維度不對，append 動作本身是 Python list 的合法操作
        try:
            self.logger.append_control_data(
                image_xy=[1, 2], # 錯誤維度，原本應為 8
                vel_cmd=[1],     # 錯誤維度
                error_list=[], 
                gain=0
            )
        except Exception as e:
            self.fail(f"Appending mismatched data caused crash: {e}")
        
        self.assertEqual(len(self.logger.image_move_data), 1)

if __name__ == '__main__':
    unittest.main()