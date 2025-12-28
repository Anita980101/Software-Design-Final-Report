import sys
sys.path.append('../utils')
sys.path.append('../utils/camera')
import time
import datetime
import numpy as np
from math import sin,cos
import keyboard
import threading
# UR5
import RTDE_func
import Rec_Cmd_func
from gen_Trajectory import *
# visual servoing
import cv2
from rs_func import get_camera
import pupil_apriltags as apriltag
# auto-gain
from Memristor_1 import Memresistor
# plotter
from draw_separate import TrajectoryPlot
current_datetime = datetime.datetime.now()
from fuzzy_class_0714 import FuzzyGainController
roc_year = current_datetime.year - 1911
formatted_datetime = current_datetime.strftime(f'{roc_year}%m%d_%H%M%S')
save_name = "ki=0.000005"

OUTPUT_FILE_NAME = './data/output/' + formatted_datetime + '_record.csv'
CONFIG_FILE_NAME = '../utils/rtde_tracking.xml'
ROBOT_HOST = '192.168.1.2'
ROBOT_PORT = 30004

################################## Record ##################################
class Recoder:
    def __init__(self,rtde,vs):
        self.i = 0
        self.ts = 0.0
        self.t_total = 0.0
        self.rtde = rtde
        self.vs = vs
        self.running = False
        self.interval = 0.01 # 10ms
        self.error = [0,0,0,0]
        self.vc = np.zeros(6)
        self.thread = threading.Thread(target=self._get_data, daemon=True) # daemon = true 當主程式結束時，它會自動被終止
    def run_thread(self):
        self.running = True
        self.thread.start()
    def renew_error(self, error):
        self.error = error
    def renew_vc(self, vc):
        self.vc = vc
    def _get_data(self):
        self.t_total = time.time()
        while self.running:
            self.ts = time.time()
            print(f"This is the {self.i} point")
            state = self.rtde.receive()
            self.renew_error(self.vs.errorSave)
            self.renew_vc(self.vs.c_v_c)
            self.vs.distance_err_thread.append(self.error)
            self.vs.command_cVc_thread.append(self.vc)
        
            if state.runtime_state == 2:
                self.vs.temp_rec_q_thread.append(state.actual_q)
                self.vs.temp_rec_dq_thread.append(state.actual_qd)
                self.vs.temp_rec_pose_thread.append(state.actual_TCP_pose)
            self.i += 1    
            time.sleep(self.interval)
            self.vs.time_thread.append(time.time()-self.ts)
            print(f"======It takes {time.time()-self.ts} (s)======") 

    def stop_thread(self):
        print(f"It takes {time.time()-self.t_total} (s)") 
        self.running = False
################################## Record ##################################

class VisualServoing:
    def __init__(self):
        self.image_moving_path = f'./data/data_1001/fuzzy/{formatted_datetime}_image_moving.csv'
        self.velocity_command_path = f'./data/data_1001/fuzzy/{formatted_datetime}_velocity_command.csv'
        self.distance_err_path = f'./data/data_1001/fuzzy/{formatted_datetime}_distance_error.csv'
        self.gain_path = f'./data/data_1001/fuzzy/{formatted_datetime}_gain.csv'
        self.memristor = f'./data/data_1001/fuzzy/{formatted_datetime}_memristor.csv'

        self.velocity_command_path_thread = f'./data/data_1001/fuzzy/{formatted_datetime}_velocity_command_thread.csv'
        self.distance_err_path_thread = f'./data/data_1001/fuzzy/{formatted_datetime}_distance_error_thread.csv'
        self.time_thread_path = f'./data/data_1001/fuzzy/{formatted_datetime}_time.csv'

        self.b_v_c = np.zeros(6)
        self.c_v_c = np.zeros(6)
        self.command_cVc = []
        self.image_move = []
        self.distance_err = []
        ########## thread ##########
        self.temp_rec_q_thread = []
        self.temp_rec_dq_thread = []
        self.temp_rec_pose_thread = []
        self.distance_err_thread = []
        self.command_cVc_thread = []
        self.time_thread = []

        self.kp = 0.01 #0.5 #1 #0.25 #0.01
        self.ki = 0.0000025
        self.errorSum = np.zeros(8)
        self.gains = []
        self.gain_c= Memresistor(0.00015,10000) #0.00016極限，再往上會明顯震動，最大gain1.0127
        self.temp_rec_q = []
        self.temp_rec_dq = []
        self.temp_rec_pose = []
        self.pixel_error = np.zeros(4)
        self.i = 0
        self.ts = None
        self.count = 0
       
        # fuzzy
        self.gain_fuzzy = []
        self.t_prev = None
        self.pre_error = 0.0
        self.fuzzy_controller = None
        self.eta = np.zeros(6)
        self.phi = np.zeros(6)

        self.NoTagNumber = 10
        self.errorSave = [0,0,0,0]
        self.StopError = 5

    def data_saving(self):
        np.savetxt(self.image_moving_path, vs.image_move, delimiter=",", fmt="%.4f")
        np.savetxt(self.velocity_command_path, vs.command_cVc, delimiter=",", fmt="%.4f")
        np.savetxt(self.velocity_command_path_thread, self.command_cVc_thread, delimiter=",", fmt="%.4f")
        np.savetxt(self.distance_err_path, vs.distance_err, delimiter=",", fmt="%.4f")
        np.savetxt(self.distance_err_path_thread, self.distance_err_thread, delimiter=",", fmt="%.4f")
        np.savetxt(self.gain_path, vs.gains, delimiter=",", fmt="%.4f")

        np.savetxt(self.time_thread_path, self.time_thread, delimiter=",", fmt="%.4f")
        # np.savetxt(gain_path, vs.gain_c.total_output, delimiter=",", fmt="%.4f")
        np.savetxt(self.memristor, vs.gain_c.total_ristor, delimiter=",", fmt="%.4f")
        print("Complete saving command and path!")

    def clear(self):
        RTDE_func.set_command["V_xyz"](commandType["V_xyz"], [0,0,0,0,0,0])
        rtde.con.send(commandType["V_xyz"])

    def draw_graph(self):
        self.data_saving()
        # plotter = TrajectoryPlot(image_moving_path, velocity_command_path, distance_err_path)
        plotter = TrajectoryPlot(self.image_moving_path,self. velocity_command_path_thread, self.distance_err_path_thread)
        plotter.load_data()
        plotter.plot_all()

################################ April Tags ###############################
    def init_apriltag_detector(self, families='tag36h11'):
        apriltag_detector = apriltag.Detector(families=families)
        # print('偵測')
        return apriltag_detector
    
########################### Coordinate Transform ##########################
    def Rij(self, theta, i):  # basic rotate matrix
            ####################### DH Parameter #######################
        #                   Link 1   Link 2    Link 3     Link 4      Link 5  Link 6
        d    = np.array([ 0.089159,       0,        0,   0.10915,    0.09465, 0.0823])
        a    = np.array([        0,  -0.425, -0.39225,         0,          0,      0])
        alph = np.array([  np.pi/2,       0,        0,   np.pi/2,   -np.pi/2,      0])

        Rij = np.array([
            [cos(theta), -sin(theta)*cos(alph[i]),  sin(theta)*sin(alph[i])],
            [sin(theta),  cos(theta)*cos(alph[i]), -cos(theta)*sin(alph[i])],
            [         0,             sin(alph[i]),             cos(alph[i])]
        ])
        return Rij

    def EndEffectorToBase(self, theta):  
        # Transform matrix : from end-effector coor. to base coor.
        R_EB = np.eye(3)
        for i in range(6):  
            R_EB = R_EB @ self.Rij(theta[i], i)      
        return R_EB
    
    def CameratoEndEffector(self):
        # Transform matrix : from end-effector coor. to camera coor.
        zero_33 = np.zeros((3,3))
        # S = np.array([[      0, -0.036,  -0.05835],
        #               [  0.036,      0,   -0.0325],
        #               [0.05835, 0.0325,         0]
        # ])
        S = np.array([[      0, -0.036,  -0.0325],
                      [  0.036,      0,   0.05835],
                      [0.0325, -0.05835,         0]
        ])
        ee_R_c = np.array([[   0,  1,  0],
                          [  -1,  0,  0],
                          [   0,  0,  1]
        ])
        temp = S @ ee_R_c
        temp1 = np.hstack((ee_R_c, temp))
        temp2 = np.hstack((zero_33, ee_R_c))
        ee_R_c_T = np.vstack((temp1, temp2))
        return ee_R_c_T
    
    def CameratoBase_v(self, c_v_c):
        # transform camera velocity to base coor.
        state1 = rtde.receive()
        e_R_B = self.EndEffectorToBase(state1.actual_q)
        b_R_ee = np.array([
            [ e_R_B[0][0],  e_R_B[0][1],  e_R_B[0][2],  0,  0,  0],
            [ e_R_B[1][0],  e_R_B[1][1],  e_R_B[1][2],  0,  0,  0],
            [ e_R_B[2][0],  e_R_B[2][1],  e_R_B[2][2],  0,  0,  0],
            [  0,  0,  0,  e_R_B[0][0],  e_R_B[0][1],  e_R_B[0][2]],
            [  0,  0,  0,  e_R_B[1][0],  e_R_B[1][1],  e_R_B[1][2]],
            [  0,  0,  0,  e_R_B[2][0],  e_R_B[2][1],  e_R_B[2][2]]
        ])

        tmp_vtoCmd = b_R_ee @ self.CameratoEndEffector() @ c_v_c
        for i in range(6):
            self.b_v_c[i] = tmp_vtoCmd[i]
        return self.b_v_c

def uv_to_camera_xy(img, tag_detector, camera_obj, draw):
    # uv-plane to normalized xy-center-plane
    image_xy = np.zeros(8)
    tags = tag_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    Cu = camera_obj.rs_para.RGB_Cu
    Cv = camera_obj.rs_para.RGB_Cv
    fu = camera_obj.rs_para.RGB_fu
    fv = camera_obj.rs_para.RGB_fv
    if draw:
        for tag in tags:
            cv2.circle(img, tuple(tag.corners[0].astype(int)), 4, (255,0,0), 2) 
            cv2.circle(img, tuple(tag.corners[1].astype(int)), 4, (255,0,0), 2)
            cv2.circle(img, tuple(tag.corners[2].astype(int)), 4, (255,0,0), 2)
            cv2.circle(img, tuple(tag.corners[3].astype(int)), 4, (255,0,0), 2)
            cv2.circle(img, tuple(tag.center.astype(int)), 4, (255,0,0), 2)
            cv2.putText(img, '0', tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, '1', tuple(tag.corners[1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, '2', tuple(tag.corners[2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, '3', tuple(tag.corners[3].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1, cv2.LINE_AA)

            for i in range(4):
                for j in range(2):
                    if j == 0:
                        image_xy[2*i] = (tag.corners[i][j]-Cu)/fu
                    elif j == 1:
                        image_xy[2*i+1] = (tag.corners[i][j]-Cv)/fv
            vs.image_move.append(image_xy)           

    try:
        flag = len(tags) > 0
    except:
        flag = False
    return flag, tags, image_xy

def visual_servoing(image, Z):

    Z = 0.31
    # command_position = np.array([255.7952, 176.4032, 256.9504, 304.752,
    #                               385.8464, 302.928, 385.056, 175.1872])
    command_position = np.array([-0.1563, -0.1788, -0.1676, 0.1602,
                                  0.1724, 0.1717, 0.1823, -0.1673])
    Le = np.array([[-1/Z, 0, image[0]/Z, image[0]*image[1], -(1+image[0]**2), image[1]],
                   [0, -1/Z, image[1]/Z, 1+image[1]**2, -image[0]*image[1], -image[0]],
                   [-1/Z, 0, image[2]/Z, image[2]*image[3], -(1+image[2]**2), image[3]],
                   [0, -1/Z, image[3]/Z, 1+image[3]**2, -image[2]*image[3], -image[2]],
                   [-1/Z, 0, image[4]/Z, image[4]*image[5], -(1+image[4]**2), image[5]],
                   [0, -1/Z, image[5]/Z, 1+image[5]**2, -image[4]*image[5], -image[4]],
                   [-1/Z, 0, image[6]/Z, image[6]*image[7], -(1+image[6]**2), image[7]],
                   [0, -1/Z, image[7]/Z, 1+image[7]**2, -image[6]*image[7], -image[6]]])
    # vc = k*(Le-1)*e
    error = command_position - image
    for l in range(4):
         vs.pixel_error[l] = ((error[2*l]**2 + error[2*l+1]**2)**(1/2))*608
    vs.errorSave = [vs.pixel_error[0],vs.pixel_error[1],vs.pixel_error[2],vs.pixel_error[3]]
    vs.errorSum += sum(error)
    vs.distance_err.append(vs.errorSave)
    #####################################自動調整gain測試#####################################
    
    # pixel_error = 0.0
    # for l in range(4):
    #     pixel_error += ((error[2*l]**2 + error[2*l+1]**2)**(1/2))*608
    # avg = pixel_error/4
    # gain_new = vs.gain_c.get_output(avg)
    # vs.kp = gain_new
    # vs.gains.append(vs.kp)
    
    #####################################自動調整gain測試#####################################
    
    ##################################### Fuzzy #############################################

    t_now = time.time()
    dt = t_now - vs.t_prev
    vs.t_prev = t_now
    
    # pixel_error = 0.0
    # for l in range(4):
    #     pixel_error += ((error[2*l]**2 + error[2*l+1]**2)**(1/2))*608
    # avg = pixel_error/4
    
    sum_square = 0
    for l in range(4):
        sum_square += error[2*l]**2 + error[2*l+1]**2
    e_total = np.sqrt(sum_square)
    
    if vs.fuzzy_controller is None:
        vs.fuzzy_controller = FuzzyGainController()
    if vs.pre_error == 0.0:
        vs.pre_error = e_total

    gain_new = vs.fuzzy_controller.compute_gain(e_total, vs.pre_error, dt)
    vs.gain_fuzzy.append(gain_new)
    vs.pre_error = e_total
    vs.kp = gain_new
    vs.gains.append(vs.kp)
    
    if vs.i == 0:
        vs.eta = np.linalg.pinv(Le) @ error
        print(f"eta is: {vs.eta}")
        error_phi = 1.0
        print("成功進入")
        
    # if vs.count == 0:
    #     error_phi = 1.0
    else:
        error_phi = np.exp(-0.5 * (t_now - vs.ts)) 
        print(f"eta is: {vs.eta}")
        print(f"t_now - ts is: {(t_now-vs.ts)} (s)")
        print(f"error_phi is: {error_phi}")

    vs.phi =  vs.kp * error_phi * vs.eta
    print(vs.phi)
    
    vs.c_v_c = vs.kp * (np.linalg.pinv(Le) @ error) - vs.phi
    print(vs.c_v_c)
    vs.command_cVc.append(vs.c_v_c)
    return vs.c_v_c
    
    # ##################################### Fuzzy #############################################
    
    ######################################## P to PI ########################################
    
    # vs.errorSum = sum(vs.errorSave)      # sum of error of 4 points
    # errorSum_single = vs.errorSum/4      # single point error(average)
    # if(vs.max_error < errorSum_single):  # update max error
    #     vs.max_error = errorSum_single
    # if(errorSum_single < (vs.max_error/3) and (vs.flag == 0)):
    #     vs.flag = 1                      # set flag to PI

    # match vs.flag:
    #     case 1:
    #         # PI control
    #         print("=============open ki=============")
    #         vs.errorSum_matrix += sum(error)  # for I control
    #         vs.c_v_c = vs.kp * (np.linalg.pinv(Le) @ error) + vs.ki * (np.linalg.pinv(Le) @ vs.errorSum_matrix)
    #     case 0:
    #         # P control
    #         vs.c_v_c = vs.kp * (np.linalg.pinv(Le) @ error)

    ######################################## P to PI ########################################
    
    # # P control
    # vs.c_v_c = vs.kp * (np.linalg.pinv(Le) @ error)
    # # PI control
    # # vs.c_v_c = vs.kp * (np.linalg.pinv(Le) @ error) + vs.ki * (np.linalg.pinv(Le) @ vs.errorSum)
    # vs.command_cVc.append(vs.c_v_c)
    # return vs.c_v_c

def run_tracking():
    vs.t_prev = time.time() 
    vs.ts = time.time()
    camera_obj = get_camera()
    tag_detector = vs.init_apriltag_detector()
    ##################### open thread #####################
    recoder.run_thread()
    #######################################################
    # i = 0
    # count = 0
    while True:
        print(f"This is the {vs.i+1} point")
        # Get camera image setting
        camera_obj.set_image_frame()
        Depth_image = camera_obj.get_Depth_image()
        RGB_image = camera_obj.get_RGB_image()
        detect_flag, tags, image_xy = uv_to_camera_xy(RGB_image, tag_detector, camera_obj, draw=True)   

        # visual Servoing Algorithm
        if detect_flag:
            camera_save_point = camera_obj.get_XYZ_point(int(tags[0].center[1]), int(tags[0].center[0]), Depth_image)
            vs.c_v_c = visual_servoing(image_xy, camera_save_point[2])
            # cv2.imshow("Image", Drawed_img)
        else:
            print("No tag detect")
            vs.count += 1
            if vs.count==vs.NoTagNumber:
                print(f"連續{vs.NoTagNumber}幀沒有影像，強制停止")
                break
     

        # Velocity Command
        Cmd = vs.CameratoBase_v(vs.c_v_c)    
        RTDE_func.set_command["V_xyz"](commandType["V_xyz"], Rec_Cmd_func.ith_command["V_xyz"](Cmd))
        rtde.con.send(commandType["V_xyz"])
        vs.i += 1

        state = rtde.receive()
        if state.runtime_state == 2:
            vs.temp_rec_q.append(state.actual_q)
            vs.temp_rec_dq.append(state.actual_qd)
            vs.temp_rec_pose.append(state.actual_TCP_pose)

        if keyboard.is_pressed('esc'):
            print("Stop by esc!")
            break

        # 平均誤差小於閥值
        if np.mean(vs.errorSave) < vs.StopError:
            print(f"It takes {time.time()-vs.ts} (s)") 
            break
        # 每個點誤差皆小於閥值
        # if vs.errorSave[0] < vs.StopError and vs.errorSave[1] < vs.StopError and vs.errorSave[2] < vs.StopError and vs.errorSave[3] < vs.StopError:
        #     print(f"It takes {time.time()-vs.ts} (s)") 
        #     break

    recoder.stop_thread()
    vs.clear()
    rtde.servo.input_int_register_0 == 0
    RTDE_func.servo_off(rtde)
    vs.draw_graph() # 包含data_saving
    return vs.temp_rec_q, vs.temp_rec_dq, vs.temp_rec_pose
    
if __name__ == "__main__":
    image_moving_path = f'./data/data_1001/fuzzy/{formatted_datetime}_image_moving.csv'
    velocity_command_path = f'./data/data_1001/fuzzy/{formatted_datetime}_velocity_command.csv'
    distance_err_path = f'./data/data_1001/fuzzy/{formatted_datetime}_distance_error.csv'
    gain_path = f'./data/data_1001/fuzzy/{formatted_datetime}_gain.csv'
    memristor = f'./data/data_1001/fuzzy/{formatted_datetime}_memristor.csv'

    velocity_command_path_thread = f'./data/data_1001/fuzzy/{formatted_datetime}_velocity_command_thread.csv'
    distance_err_path_thread = f'./data/data_1001/fuzzy/{formatted_datetime}_distance_error_thread.csv'
    time_thread_path = f'./data/data_1001/fuzzy/{formatted_datetime}_time.csv'
    # cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

    rtde = RTDE_func.rtde_init(np.zeros((6, 1)), 'V_xyz', ROBOT_HOST, CONFIG_FILE_NAME)
    vs = VisualServoing()
    recoder = Recoder(rtde, vs)
    commandType = {
        "joint": rtde.set_q,
        "cartesian": rtde.set_xyz,
        "V_xyz": rtde.set_v_xyz
    }

    RTDE_func.wait_for_ready(rtde)
    RTDE_func.servo_on(rtde)
    rec_q, rec_dq, rec_pose = run_tracking()
    Rec_Cmd_func.save_data(rec_q, rec_dq, rec_pose, OUTPUT_FILE_NAME)
    

