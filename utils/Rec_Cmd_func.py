import csv
import os
import numpy as np
import json

class Joint_Cmd():
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    q6 = []
    dq1 = []
    dq2 = []
    dq3 = []
    dq4 = []
    dq5 = []
    dq6 = []

class Joint_Rec():
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    q6 = []
    dq1 = []
    dq2 = []
    dq3 = []
    dq4 = []
    dq5 = []
    dq6 = []

class XYZ_Cmd():
    X = []
    Y = []
    Z = []
    alpha = []
    beta  = []
    gamma = []

class v_XYZ_Cmd():
    vX = []
    vY = []
    vZ = []
    rX = []
    rY = []
    rZ = []

class State_Record():
    def __init__(self):
        self.sampT = 0.008
        self.y_demo = None
        self.dy_demo = None
        self.ddy_demo = None
        self.y_reproduce = None
        self.dy_reproduce = None
        self.ddy_reproduce = None
        self.Bs = None
        self.taus = None
        self.coupling_terms = None

    def save_to_file(self, filename):
        data = {
            'y_demo': self.y_demo.tolist() if self.y_demo is not None else None,
            'dy_demo': self.dy_demo.tolist() if self.dy_demo is not None else None,
            'ddy_demo': self.ddy_demo.tolist() if self.ddy_demo is not None else None,
            'y_reproduce': self.y_reproduce.tolist() if self.y_reproduce is not None else None,
            'dy_reproduce': self.dy_reproduce.tolist() if self.dy_reproduce is not None else None,
            'ddy_reproduce': self.ddy_reproduce.tolist() if self.ddy_reproduce is not None else None,
            'Bs': self.Bs.tolist() if self.Bs is not None else None,
            'taus': self.taus.tolist() if self.taus is not None else None,
            'coupling_terms': self.coupling_terms.tolist() if self.coupling_terms is not None else None
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, filename):
        record = self
        with open(filename, 'r') as f:
            data = json.load(f)
            record.y_demo = np.array(data['y_demo'])
            record.dy_demo = np.array(data['dy_demo'])
            record.ddy_demo = np.array(data['ddy_demo'])
            record.y_reproduce = np.array(data['y_reproduce'])
            record.dy_reproduce = np.array(data['dy_reproduce'])
            record.ddy_reproduce = np.array(data['ddy_reproduce'])
            record.Bs = np.array(data['Bs'])
            record.taus = np.array(data['taus'])
            record.coupling_terms = np.array(data['coupling_terms'])
        return record

def read_command_joint(INPUT_FILE_NAME):
    temp_cmd = Joint_Cmd
    with open(INPUT_FILE_NAME, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            temp_cmd.q1.append(float(row[0]))
            temp_cmd.q2.append(float(row[1]))
            temp_cmd.q3.append(float(row[2]))
            temp_cmd.q4.append(float(row[3]))
            temp_cmd.q5.append(float(row[4]))
            temp_cmd.q6.append(float(row[5]))
    return temp_cmd, int(len(temp_cmd.q1))

def read_command_xyz(INPUT_FILE_NAME):
    temp_cmd = XYZ_Cmd
    with open(INPUT_FILE_NAME, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            temp_cmd.X.append(float(row[0]))
            temp_cmd.Y.append(float(row[1]))
            temp_cmd.Z.append(float(row[2]))
            temp_cmd.alpha.append(float(row[3]))
            temp_cmd.beta.append(float(row[4]))
            temp_cmd.gamma.append(float(row[5]))
    return temp_cmd, int(len(temp_cmd.X))

def read_command_v_xyz_fix(INPUT_FILE_NAME):
    temp_cmd = v_XYZ_Cmd
    with open(INPUT_FILE_NAME, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            temp_cmd.vX.append(float(row[0]))
            temp_cmd.vY.append(float(row[1]))
            temp_cmd.vZ.append(float(row[2]))
            temp_cmd.rX.append(float(row[3]))
            temp_cmd.rY.append(float(row[4]))
            temp_cmd.rZ.append(float(row[5]))
    return temp_cmd, int(len(temp_cmd.vX))

def read_command_v_xyz(INPUT_FILE_NAME):
    temp_cmd = v_XYZ_Cmd
    with open(INPUT_FILE_NAME, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            temp_cmd.vX.append(float(row[0]))
            temp_cmd.vY.append(float(row[1]))
            temp_cmd.vZ.append(float(row[2]))
            temp_cmd.rX.append(float(row[3]))
            temp_cmd.rY.append(float(row[4]))
            temp_cmd.rZ.append(float(row[5]))
    return temp_cmd, int(len(temp_cmd.vX))

read_command = {
    'joint' : read_command_joint,
    'cartesian' : read_command_xyz,
    'V_xyz' : read_command_v_xyz,
    'V_xyz_fix' : read_command_v_xyz_fix
}

def ith_joint(Cmd, i):
    return [Cmd.q1[i], Cmd.q2[i], Cmd.q3[i], Cmd.q4[i], Cmd.q5[i], Cmd.q6[i]]

def ith_xyz(Cmd, i):
    return [Cmd.X[i], Cmd.Y[i], Cmd.Z[i], Cmd.alpha[i], Cmd.beta[i], Cmd.gamma[i]]

def ith_v_xyz_fix(Cmd, i):
    return [Cmd.vX[i], Cmd.vY[i], Cmd.vZ[i], Cmd.rX[i], Cmd.rY[i], Cmd.rZ[i]]

def ith_v_xyz(Cmd):  ##配合視覺伺服格式
    return Cmd

ith_command = {
    'joint' : ith_joint,
    'cartesian' : ith_xyz,
    'V_xyz' : ith_v_xyz,
    'V_xyz_fix' : ith_v_xyz_fix
}

def save_command_XYZ(temp_cmd, COMMAND_FILE_NAME):
    cmd = XYZ_Cmd()
    for i in range(len(temp_cmd)):
        cmd.X.append(temp_cmd[i][0])
        cmd.Y.append(temp_cmd[i][1])
        cmd.Z.append(temp_cmd[i][2])
        cmd.alpha.append(temp_cmd[i][3])
        cmd.beta.append(temp_cmd[i][4])
        cmd.gamma.append(temp_cmd[i][5])

    with open(COMMAND_FILE_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(temp_cmd)):
            writer.writerow([cmd.X[i],   cmd.Y[i],   cmd.Z[i], \
                             cmd.alpha[i],   cmd.beta[i],   cmd.gamma[i]])

def save_data(rec_q, rec_dq, rec_pose, OUTPUT_FILE_NAME):
    with open(OUTPUT_FILE_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(rec_q)):
            writer.writerow([   rec_q[i][0],    rec_q[i][1],    rec_q[i][2],
                                rec_q[i][3],    rec_q[i][4],    rec_q[i][5],
                               rec_dq[i][0],   rec_dq[i][1],   rec_dq[i][2],
                               rec_dq[i][3],   rec_dq[i][4],   rec_dq[i][5],
                             rec_pose[i][0], rec_pose[i][1], rec_pose[i][2],
                             rec_pose[i][3], rec_pose[i][4], rec_pose[i][5]])

