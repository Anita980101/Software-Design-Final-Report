import sys
sys.path.append('../utils')
sys.path.append('../utils/camera')
import time
import numpy as np
import keyboard
import cv2
import traceback
import RTDE_func
import Rec_Cmd_func
from rs_func import get_camera 
import pupil_apriltags as apriltag

from config import Config
from data_logger import DataLogger, AsyncRecorder
from kinematics import RobotKinematics
from fuzzy_class_0714 import FuzzyGainController
from memristor import Memristor
from visual_servoing_controller import VisualServoController
from draw import TrajectoryPlot

def uv_to_normalized_xy(img, detector, camera_obj):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    
    features = np.zeros(8)
    detected = False
    
    if len(tags) > 0:
        detected = True
        tag = tags[0] 
        corners = tag.corners
        
        cu = camera_obj.rs_para.RGB_Cu
        cv = camera_obj.rs_para.RGB_Cv
        fu = camera_obj.rs_para.RGB_fu
        fv = camera_obj.rs_para.RGB_fv
        
        for i in range(4):
            features[2*i]   = (corners[i][0] - cu) / fu
            features[2*i+1] = (corners[i][1] - cv) / fv
            
    return detected, tags, features

def main():
    paths = Config.get_paths()
    logger = DataLogger(paths)
    
    fuzzy = FuzzyGainController()
    memristor = Memristor(m_value=Config.MEM_M_VALUE, saturation=Config.MEM_SATURATION)
    
    controller = VisualServoController(fuzzy_ctrl=fuzzy, memristor_ctrl=memristor)
    
    # UR5 機械手臂連線
    print("Connecting to UR5")
    rtde = RTDE_func.rtde_init(np.zeros((6, 1)), 'V_xyz', Config.ROBOT_HOST, Config.CONFIG_FILE)
    commandType = {"V_xyz": rtde.set_v_xyz}
    
    print("Initializing Realsense")
    camera_obj = get_camera() 
    detector = apriltag.Detector(families='tag36h11')
    
    # 開啟 Thread
    recorder = AsyncRecorder(rtde, logger)

    print(f"System Ready. Mode: {Config.CONTROL_MODE}")
    RTDE_func.wait_for_ready(rtde)
    RTDE_func.servo_on(rtde)
    recorder.start()

    no_tag_count = 0
    
    try:
        while True:
            camera_obj.set_image_frame() 
            rgb_img = camera_obj.get_RGB_image()
            
            is_detected, tags, features = uv_to_normalized_xy(rgb_img, detector, camera_obj)

            if not is_detected:
                print("No tag detected.")
                no_tag_count += 1
                if no_tag_count >= Config.NO_TAG_LIMIT:
                    print(f"Lost tag for {Config.NO_TAG_LIMIT} frames. Stopping.")
                    break
                continue
            
            no_tag_count = 0 
            
            c_v_c, pixel_errors, current_gain, mem_val = controller.update(features)
            
            recorder.update_state(pixel_errors, c_v_c)
            logger.append_control_data(features, c_v_c, pixel_errors, current_gain, memristor_val=mem_val)

            state = rtde.receive()
            current_q = state.actual_q
            
            base_velocity_cmd = RobotKinematics.camera_velocity_to_base(c_v_c, current_q)
            
            RTDE_func.set_command["V_xyz"](commandType["V_xyz"], Rec_Cmd_func.ith_command["V_xyz"](base_velocity_cmd))
            rtde.con.send(commandType["V_xyz"])

            avg_error = np.mean(pixel_errors)
            print(f"Mode: {Config.CONTROL_MODE} | Err: {avg_error:.2f} | Gain: {current_gain:.5f}")
            
            if avg_error < Config.STOP_ERROR_THRESHOLD:
                print("Target Reached!")
                break
                
            if keyboard.is_pressed('esc'):
                print("Stopped by User (ESC).")
                break

    except Exception as e:
        print("Runtime Error Occurred:")
        traceback.print_exc()
        
    finally:
        print("Stopping system...")
        recorder.stop()
        
        # 停止機械手臂
        RTDE_func.set_command["V_xyz"](commandType["V_xyz"], [0]*6)
        rtde.con.send(commandType["V_xyz"])
        RTDE_func.servo_off(rtde)
        
        # 儲存數據
        logger.save_all_to_csv()
        
        # 畫圖
        print("Generating plots")
        try:
            plotter = TrajectoryPlot(
                image_traj_path=paths['image_moving'], 
                velocity_path=paths['vel_cmd_thread'], 
                dist_err_path=paths['dist_err_thread'],
                output_dir=paths['base_dir']
            )
            plotter.load_data()
            plotter.plot_all()
            print(f"Plots saved to {paths['base_dir']}")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()