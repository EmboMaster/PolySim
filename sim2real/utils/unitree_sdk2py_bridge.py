import mujoco
import numpy as np
import glfw
import sys
import struct
from termcolor import colored
from loguru import logger
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from sim2real.utils.robot import Robot

import pygame

class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data, robot_config):
        self.robot = Robot(robot_config)
        robot_type = self.robot.ROBOT_TYPE
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.free_base = robot_config["FREE_BASE"]
        if self.free_base:
            self.num_motor = self.mj_model.nu - 6
        else: self.num_motor = self.mj_model.nu

        self.torques = np.zeros(self.num_motor)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)

        self.use_sensor = self.robot.USE_SENSOR # True: use sensor data; False: use ground truth data
        # Check if the robot is using sensor data
        self.have_imu_ = False
        self.have_frame_sensor_ = False
        if self.use_sensor:
            MOTOR_SENSOR_NUM = 3
            self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
            # Check sensor
            for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
                name = mujoco.mj_id2name(
                    self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
                )
                if name == "imu_quat":
                    self.have_imu_ = True
                if name == "frame_pos":
                    self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

    def LowCmdHandler(self, msg):
        self.low_cmd = msg

    def PublishLowState(self):
        if self.mj_data != None:
            if self.use_sensor:
                for i in range(self.num_motor):
                    self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                    self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                        i + self.num_motor
                    ]
                    self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                        i + 2 * self.num_motor
                    ]
                    # TODO: temperature (Default: human body temperature :))
                    # self.low_state.temperature[0] = 37.5
                    # self.low_state.temperature[1] = 37.5
            else:
                for i in range(self.num_motor):
                    self.low_state.motor_state[i].q = self.mj_data.qpos[7+i]
                    self.low_state.motor_state[i].dq = self.mj_data.qvel[6+i]
                    self.low_state.motor_state[i].ddq = self.mj_data.qacc[6+i]
                    self.low_state.motor_state[i].tau_est = self.mj_data.actuator_force[i]
                    # TODO: temperature (Default: human body temperature :))
                    # self.low_state.temperature[0] = 37.5
                    # self.low_state.temperature[1] = 37.5

            # Get data from sensors
            if self.use_sensor and self.have_frame_sensor_:

                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 3
                ]
                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 6
                ]
            else:
                # Get data from ground truth
                # quaternion: w, x, y, z
                self.low_state.imu_state.quaternion[0] = self.mj_data.qpos[3]
                self.low_state.imu_state.quaternion[1] = self.mj_data.qpos[4]
                self.low_state.imu_state.quaternion[2] = self.mj_data.qpos[5]
                self.low_state.imu_state.quaternion[3] = self.mj_data.qpos[6]
                # angular velocity: x, y, z
                self.low_state.imu_state.gyroscope[0] = self.mj_data.qvel[3]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.qvel[4]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.qvel[5]
            # acceleration: x, y, z (only available when frame sensor is enabled)
            if self.have_frame_sensor_:
                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 9
                ]
            self.low_state.tick = int(self.mj_data.time*1e3)
            self.low_state_puber.Write(self.low_state)

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # Yuanhang: may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                logger.info(f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}")
            index = index + self.mj_model.sensor_dim[i]
        print(" ")

class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
