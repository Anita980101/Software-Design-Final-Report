import unittest
import sys
import os
from unittest.mock import patch

# 將上層目錄加入 path 以便 import 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

class TestConfig(unittest.TestCase):
    
    def test_constants_exist(self):
        """測試必要的常數是否存在"""
        self.assertTrue(hasattr(Config, 'ROBOT_HOST'))
        self.assertTrue(hasattr(Config, 'COMMAND_POSITION'))
        self.assertIsInstance(Config.COMMAND_POSITION, list)
        self.assertEqual(len(Config.COMMAND_POSITION), 8)

    @patch('os.makedirs')
    def test_get_paths(self, mock_makedirs):
        """測試路徑生成字典結構是否正確"""
        test_dir = './test_data'
        paths = Config.get_paths(base_dir=test_dir)
        
        # 驗證是否呼叫了建立資料夾
        self.assertTrue(mock_makedirs.called)
        required_keys = [
            'image_moving', 'velocity_cmd', 'distance_err', 
            'gain', 'memristor', 'vel_cmd_thread', 'output_record'
        ]
        for key in required_keys:
            self.assertIn(key, paths)
            
        # 驗證路徑字串是否包含 base_dir
        self.assertIn(test_dir, paths['image_moving'])

if __name__ == '__main__':
    unittest.main()