import numpy as np
from math import sin, cos

class RobotKinematics:
    def get_Rij(theta, i):

        # DH 表
        alph = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        return np.array([
            [cos(theta), -sin(theta)*cos(alph[i]),  sin(theta)*sin(alph[i])],
            [sin(theta),  cos(theta)*cos(alph[i]), -cos(theta)*sin(alph[i])],
            [0,           sin(alph[i]),             cos(alph[i])]
        ])

    def end_effector_to_base(theta_list):
        R_EB = np.eye(3)
        for i in range(6):
            R_EB = R_EB @ RobotKinematics.get_Rij(theta_list[i], i)
        return R_EB

    def camera_to_end_effector():
        zero_33 = np.zeros((3, 3))

        # 相機外部參數
        S = np.array([[0, -0.036, -0.0325],
                      [0.036, 0, 0.05835],
                      [0.0325, -0.05835, 0]])
        
        ee_R_c = np.array([[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 1]])
        
        temp = S @ ee_R_c
        top = np.hstack((ee_R_c, temp))
        bottom = np.hstack((zero_33, ee_R_c))
        return np.vstack((top, bottom))

    def camera_velocity_to_base(c_v_c, current_q):
        e_R_B = RobotKinematics.end_effector_to_base(current_q)
        
        b_R_ee_top = np.hstack((e_R_B, np.zeros((3,3))))
        b_R_ee_bot = np.hstack((np.zeros((3,3)), e_R_B))
        b_R_ee = np.vstack((b_R_ee_top, b_R_ee_bot))

        velocity_cmd = b_R_ee @ RobotKinematics.camera_to_end_effector() @ c_v_c
        return velocity_cmd