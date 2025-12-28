import rtdeState
import time

def list_to_set_q(set_q, list):
    for i in range(0, len(list)):
        set_q.__dict__["input_double_register_%i" % i] = list[i]
    return set_q


def list_to_set_xyz(set_xyz, list):
    for i in range(0, len(list)):
        set_xyz.__dict__["input_double_register_%i" % (i+6)] = list[i]
    return set_xyz

def list_to_set_v_xyz(set_v_xyz, list):
    for i in range(0, len(list)):
        set_v_xyz.__dict__["input_double_register_%i" % (i+12)] = list[i]
    return set_v_xyz

def list_to_set_v_xyz_fix(set_v_xyz_fix, list):
    for i in range(0, len(list)):
        set_v_xyz_fix.__dict__["input_double_register_%i" % (i+12)] = list[i]
    return set_v_xyz_fix

set_command = {
    'joint' :  list_to_set_q,
    'cartesian' : list_to_set_xyz,
    'V_xyz' : list_to_set_v_xyz,
    'V_xyz_fix' : list_to_set_v_xyz_fix
}

def rtde_init(INI_VALUE, COMMAND_TYPE, ROBOT_HOST, CONFIG_FILE_NAME):
    print("Initializing RTDE")
    rtde = rtdeState.RtdeState(ROBOT_HOST, CONFIG_FILE_NAME)
    rtde.initialize()
    if COMMAND_TYPE=='cartesian':
        set_command[COMMAND_TYPE](rtde.set_xyz, INI_VALUE)
        rtde.con.send(rtde.set_xyz)
    elif COMMAND_TYPE=='joint':
        set_command[COMMAND_TYPE](rtde.set_q, INI_VALUE)
        rtde.con.send(rtde.set_q)
    elif COMMAND_TYPE=='V_xyz':
        set_command[COMMAND_TYPE](rtde.set_v_xyz, INI_VALUE)
        rtde.con.send(rtde.set_v_xyz)
    rtde.servo.input_int_register_0 = 0
    print("Success")
    return rtde

def wait_for_ready(rtde):
    print("Waiting for ready")
    # Wait for program to be started and ready.
    state = rtde.receive()
    while state.output_int_register_0 != 1:
        # print(state.output_int_register_0)
        state = rtde.receive()

def servo_on(rtde):
    # Send command to robot to begin servoing.
    rtde.servo.input_int_register_0 = 1
    rtde.con.send(rtde.servo)
    print("Servo on")
    time.sleep(0.01)

def servo_off(rtde):
    # Stop servoing.
    rtde.servo.input_int_register_0 = 0
    rtde.con.send(rtde.servo)
    time.sleep(0.01)
    rtde.con.send_pause()
    rtde.con.disconnect()
    print("Servo off")

def wait_for_done(rtde):
    # print("Waiting for done")
    # Wait for program to be started and ready.
    state = rtde.receive()
    while state.output_int_register_2 != 0:
        state = rtde.receive() 
   
def wait_for_go(rtde):
    # print("Waiting for new")
    # Wait for program to be started and ready.
    state = rtde.receive()
    while state.output_int_register_2 != 1:
        state = rtde.receive() 