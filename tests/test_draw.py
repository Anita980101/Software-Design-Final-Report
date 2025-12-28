import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# 設定路徑以 import 上層模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draw import TrajectoryPlot

class TestTrajectoryPlot(unittest.TestCase):
    def setUp(self):
        # 建立一個測試用的輸出目錄
        self.output_dir = './test_output'
        self.plotter = TrajectoryPlot('img.csv', 'vel.csv', 'err.csv', self.output_dir)

    @patch('numpy.loadtxt')
    def test_load_data(self, mock_loadtxt):
        """測試資料讀取邏輯"""
        # 模擬回傳一個 10列 8行的數據 (模擬 4 個特徵點的 (u,v))
        mock_data = np.zeros((10, 8))
        mock_loadtxt.return_value = mock_data
        
        self.plotter.load_data()
        
        self.assertIsNotNone(self.plotter.data)
        self.assertIsNotNone(self.plotter.start_point)
        self.assertIsNotNone(self.plotter.end_point)

    # 這裡加入 patch('draw.Config') 來模擬 Config
    @patch('draw.Config') 
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_execution(self, mock_savefig, mock_figure, MockConfig):
        """測試繪圖函式是否順利執行 (Mock pyplot 和 Config)"""
        
        # 1. 設定 Mock Config 的行為
        # 當 draw.py 呼叫 get_goal_points_for_plot() 時，回傳假的目標點列表
        # 格式是一組 (x, y) 的 list
        MockConfig.get_goal_points_for_plot.return_value = [
            (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)
        ]

        # 2. 注入假數據給 plotter (避免 load_data 失敗導致 data 為 None)
        self.plotter.data = np.zeros((5, 8))
        self.plotter.start_point = [(0,0)]*4
        self.plotter.end_point = [(0,0)]*4
        
        # 3. 執行繪圖 (現在 Config 有了這個函式，就不會報錯了)
        self.plotter._plot_trajectory()
        
        # 4. 驗證是否呼叫了繪圖指令
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_savefig.called)

    @patch('numpy.loadtxt')
    def test_load_data_empty_file(self, mock_loadtxt):
        """測試：讀取到的檔案是空的或格式錯誤"""
        mock_loadtxt.side_effect = Exception("Empty file")
        
        try:
            self.plotter.load_data()
        except Exception:
            self.fail("load_data should handle exceptions internally")
            
        self.assertIsNone(self.plotter.data)

    def test_remove_leading_zeros(self):
        """測試：去除前導零的功能"""
        raw_data = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        cleaned = self.plotter._remove_leading_zeros(raw_data)
        
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0, 0], 1)

if __name__ == '__main__':
    unittest.main()