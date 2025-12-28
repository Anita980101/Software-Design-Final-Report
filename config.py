import datetime
import os

class Config:
    # 機械手臂連線設定
    ROBOT_HOST = '192.168.1.2'
    ROBOT_PORT = 30004
    CONFIG_FILE = '../utils/rtde_tracking.xml'

    # 模式選擇：'P', 'PI', 'FUZZY', 'MEMRISTOR'
    CONTROL_MODE = 'MEMRISTOR'
    
    # 控制參數
    Z_HEIGHT = 0.31
    KP_DEFAULT = 0.01
    KI_DEFAULT = 0.0000025
    STOP_ERROR_THRESHOLD = 5.0
    NO_TAG_LIMIT = 10
    MEM_M_VALUE = 0.00015   
    MEM_SATURATION = 10000.0 
    
    # Visual Servoing 目標位置
    COMMAND_POSITION = [-0.1563, -0.1788, -0.1676, 0.1602,
                        0.1724, 0.1717, 0.1823, -0.1673]

    def get_paths(base_dir='./data/data_1001/fuzzy'):
        current_datetime = datetime.datetime.now()
        roc_year = current_datetime.year - 1911
        fmt_time = current_datetime.strftime(f'{roc_year}%m%d_%H%M%S')
        os.makedirs(base_dir, exist_ok=True)
        return {
            'base_dir': base_dir,
            'image_moving': f'{base_dir}/{fmt_time}_image_moving.csv',
            'velocity_cmd': f'{base_dir}/{fmt_time}_velocity_command.csv',
            'distance_err': f'{base_dir}/{fmt_time}_distance_error.csv',
            'gain': f'{base_dir}/{fmt_time}_gain.csv',
            'memristor': f'{base_dir}/{fmt_time}_memristor.csv',
            'vel_cmd_thread': f'{base_dir}/{fmt_time}_velocity_command_thread.csv',
            'dist_err_thread': f'{base_dir}/{fmt_time}_distance_error_thread.csv',
            'time_thread': f'{base_dir}/{fmt_time}_time.csv',
            'output_record': f'./data/output/{fmt_time}_record.csv'
        }
    
    def get_goal_points_for_plot():
        cmd = Config.COMMAND_POSITION
        return [(cmd[2*i], cmd[2*i+1]) for i in range(4)]

