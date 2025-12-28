import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draw_separate import TrajectoryPlot

class TestDrawLegacy(unittest.TestCase):
    def setUp(self):
        self.plotter = TrajectoryPlot('img.csv', 'vel.csv', 'err.csv')

    @patch('numpy.loadtxt')
    def test_load_data(self, mock_loadtxt):
        """測試資料載入與解析 (Legacy)"""
        # 模擬舊版需要的數據格式 (至少要有多列數據)
        # 舊版直接取 data[1, 0] 等硬編碼索引，所以需要足夠的資料
        mock_data = np.zeros((10, 8)) 
        mock_loadtxt.return_value = mock_data
        
        self.plotter.load_data()
        
        self.assertIsNotNone(self.plotter.data)
        self.assertIsNotNone(self.plotter.start_point)
        self.assertIsNotNone(self.plotter.end_point)

    def test_remove_leading_zeros(self):
        """測試去除前導零"""
        data = np.array([
            [0,0,0,0],
            [0,0,0,0],
            [1,2,3,4],
            [5,6,7,8]
        ])
        cleaned = self.plotter._remove_leading_zeros(data)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0,0], 1)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('numpy.loadtxt')
    def test_plot_all(self, mock_loadtxt, mock_savefig, mock_show, mock_figure):
        """測試繪圖流程"""
        # 設定假數據
        mock_loadtxt.return_value = np.zeros((10, 8))
        self.plotter.load_data()
        
        # 執行繪圖
        self.plotter.plot_all()
        
        # 舊版會畫 4 張圖，檢查 figure 是否被呼叫至少 4 次
        self.assertGreaterEqual(mock_figure.call_count, 4)

if __name__ == '__main__':
    unittest.main()