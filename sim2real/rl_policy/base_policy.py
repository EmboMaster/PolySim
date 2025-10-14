import rclpy
from rclpy.node import Node
import numpy as np
import time
import threading
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
# import torch
import pygame
# from pynput import keyboard
from sshkeyboard import listen_keyboard
from termcolor import colored
import onnxruntime
# import ipdb; ipdb.set_trace()
import sys
import loguru
sys.path.append('../')
sys.path.append('./')
from std_msgs.msg import Float64MultiArray, Bool
from sim2real.utils.robot import Robot
from sim2real.utils.history_handler import HistoryHandler
from sim2real.utils.state_processor import StateProcessor
from sim2real.utils.command_sender import CommandSender
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

class BasePolicy:
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        self.config = config
        self.robot = Robot(config)
        self.node = node
        self.rate = self.node.create_rate(rl_rate)
        self.robot_state_data = None
        self.use_mocap = config.get("use_mocap", False)
        if self.use_mocap:
            # Subscribers
            self.node.create_subscription(Odometry, '/odometry', self.odometry_callback, 10)
        if config.get("INTERFACE", None):
            ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
        else:
            ChannelFactoryInitialize(config["DOMAIN_ID"])
        
        self.state_processor = StateProcessor(config)
        self.command_sender = CommandSender(config)

        self.setup_policy(model_path, use_jit)

        self.num_dofs = self.robot.NUM_JOINTS
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.default_dof_angles = self.robot.DEFAULT_DOF_ANGLES
        self.policy_action_scale = policy_action_scale

        # Keypress control state
        self.use_policy_action = False

        self.period = 1.0 / rl_rate  # Calculate period in seconds
        self.last_time = time.time()

        self.decimation = decimation

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[0.78]])
        
        self.motor_pos_lower_limit_list = self.config.get("motor_pos_lower_limit_list", None)
        self.motor_pos_upper_limit_list = self.config.get("motor_pos_upper_limit_list", None)
        self.motor_vel_limit_list = self.config.get("motor_vel_limit_list", None)
        self.motor_effort_limit_list = self.config.get("motor_effort_limit_list", None)
        
        self.use_history = self.config["USE_HISTORY"]
        self.obs_scales = self.config["obs_scales"]
        self.history_handler = None
        self.current_obs = None
        if self.use_history: 
            self.history_handler = HistoryHandler(self.config["history_config"], self.config["obs_dims"])
            self.current_obs = {key: np.zeros((1, self.config["obs_dims"][key])) for key in self.config["obs_dims"].keys()}

        if self.config.get("USE_JOYSTICK", False):
            # Yuanhang: pygame event can only run in main thread on Mac, so we need to implement it with rl inference
            print("Using joystick")
            self.use_joystick = True
            self.key_states = {}
            self.last_key_states = {}
            self.wireless_controller_subscriber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
            self.wireless_controller_subscriber.Init(self.WirelessControllerHandler, 1)
            self.wc_msg = None
            self.wc_key_map = {
                1: "R1",
                2: "L1",
                3: "L1+R1",
                4: "start",
                8: "select",
                16: "R2",
                32: "L2",
                64: "F1", # not used in sim2sim
                128: "F2", # not used in sim2sim
                256: "A",
                512: "B",
                768: "A+B",
                1024: "X",
                1280: "A+X",
                2048: "Y",
                2304: "A+Y",
                2560: "B+Y",
                3072: "X+Y",
                4096: "up",
                4608: "B+up",
                8192: "right",
                8448: "A+right",
                10240: "Y+right",
                16384: "down",
                16896: "B+down",
                32768: "left",
                33024: "A+left",
                34816: "Y+left",
            }
            print("Wireless Controller Initialized")
        else:
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
            self.key_listener_thread.start()
        self.elastic_band_disabled = False

    def disable_elastic_band(self):
        if not self.elastic_band_disabled:
            msg = Bool()
            msg.data = True
            self.disable_elastic_band_pub.publish(msg)
            self.elastic_band_disabled = True
            loguru.logger.info("Sent disable elastic band signal")

    def enable_elastic_band(self):
        if self.elastic_band_disabled:
            msg = Bool()
            msg.data = False
            self.disable_elastic_band_pub.publish(msg)
            self.elastic_band_disabled = False
            loguru.logger.info("Sent enable elastic band signal")

    def WirelessControllerHandler(self, msg: WirelessController_):
        self.wc_msg = msg

    def setup_policy(self, model_path, use_jit):
        # load onnx policy
        if not use_jit:
            self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
            def policy_act(obs):
                return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        else:
            self.jit_policy = torch.jit.load(model_path)
            def policy_act(obs):
                obs = torch.tensor(obs)
                action_10dof = self.jit_policy(obs)
                action_19dof = torch.cat([action_10dof, torch.zeros(1, 9)], dim=1)
                return action_19dof.detach().numpy()
        self.policy = policy_act

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state_data [:3]: robot base pos
        # robot_state_data [3:7]: robot base quaternion
        # robot_state_data [7:7+dof_num]: joint angles 
        # robot_state_data [7+dof_num: 7+dof_num+3]: base linear velocity
        # robot_state_data [7+dof_num+3: 7+dof_num+6]: base angular velocity
        # robot_state_data [7+dof_num+6: 7+dof_num+6+dof_num]: joint velocities
        raise NotImplementedError


    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        if self.get_ready_state:
            # interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        else:
            return dof_pos

    def _get_obs_history(self,):
        assert "history_config" in self.config.keys()
        history_config = self.config["history_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)
    
    def get_policy_action(self, robot_state_data):
        # Process low states
        obs = self.prepare_obs_for_rl(robot_state_data)
        # Policy inference
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()  
        scaled_policy_action = policy_action * self.policy_action_scale

        return scaled_policy_action

    def rl_inference(self):
        # Get states
        self.robot_state_data = self.state_processor._prepare_low_state()
        if self.robot_state_data is None:
            print("No robot state data received, skipping rl inference")
            return
        # Get policy action
        scaled_policy_action = self.get_policy_action(self.robot_state_data)
        # print("scaled_policy_action: ",scaled_policy_action.shape)
        # print("scaled_policy_action 1: ",scaled_policy_action)
        if self.get_ready_state:
            # import ipdb; ipdb.set_trace()
            q_target = self.get_init_target(self.robot_state_data)
            if self.init_count > 500:
                self.init_count = 500
        elif not self.use_policy_action:
            # print("1")
            q_target = self.robot_state_data[:, 7:7+self.num_dofs]
        else:
            # print("2")
            if not scaled_policy_action.shape[1] == self.num_dofs:
                # print("4")
                scaled_policy_action = np.concatenate([scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1)
            q_target = scaled_policy_action + self.default_dof_angles
            # print("3")
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # print("cmd_q: ",cmd_q.shape)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

    def handle_keyboard_button(self, keycode):
        if keycode == "]":
            self.disable_elastic_band()
            self.use_policy_action = True
            self.get_ready_state = False
            self.node.get_logger().info("Using policy actions")
            # self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            self.phase = 0.0
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            self.node.get_logger().info("Actions set to zero")
        elif keycode == "i":
            self.enable_elastic_band()
            self.get_ready_state = True
            self.init_count = 0
            self.node.get_logger().info("Setting to init state")
        elif keycode == "w" and self.stand_command:
            self.lin_vel_command[0, 0]+=0.1
        elif keycode == "s" and self.stand_command:
            self.lin_vel_command[0, 0]-=0.1
        elif keycode == "a" and self.stand_command:
            self.lin_vel_command[0, 1]+=0.1 
        elif keycode == "d" and self.stand_command:
            self.lin_vel_command[0, 1]-=0.1
        elif keycode == "q":
            self.ang_vel_command[0, 0]-=0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0]+=0.1
        elif keycode == "z":
            self.ang_vel_command[0, 0] = 0.
            self.lin_vel_command[0, 0] = 0.
            self.lin_vel_command[0, 1] = 0.
        elif keycode == "1":
            self.base_height_command += 0.05
        elif keycode == "2":
            self.base_height_command -= 0.05
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "=":
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
        print(f"Linear velocity command: {self.lin_vel_command}")
        print(f"Angular velocity command: {self.ang_vel_command}")
        print(f"Base height command: {self.base_height_command}")
        print(f"Stand command: {self.stand_command}")

    def process_joystick_input(self):
        # Process stick
        if self.wc_msg.keys == 0:
            self.lin_vel_command[0, 1] = -(self.wc_msg.lx if abs(self.wc_msg.lx) > 0.1 else 0.0) * self.stand_command[0, 0] * 0.5
            self.lin_vel_command[0, 0] = (self.wc_msg.ly if abs(self.wc_msg.ly) > 0.1 else 0.0) * self.stand_command[0, 0] * 0.5
            self.ang_vel_command[0, 0] = -(self.wc_msg.rx if abs(self.wc_msg.rx) > 0.1 else 0.0) * self.stand_command[0, 0]
        cur_key = self.wc_key_map.get(self.wc_msg.keys, None)
        self.last_key_states = self.key_states.copy()
        if cur_key:
            self.key_states[cur_key] = True
        else:
            self.key_states = {key: False for key in self.wc_key_map.values()}
        
        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)

    def handle_joystick_button(self, cur_key):
        # Handle button press
        if cur_key == "start":
            self.history_handler.reset([0])
            self.use_policy_action = True
            self.get_ready_state = False
            self.node.get_logger().info(colored("Using policy actions", "blue"))
            self.phase = 0.0
        elif cur_key == "B+Y":
            self.use_policy_action = False
            self.get_ready_state = False
            self.node.get_logger().info(colored("Actions set to zero", "blue"))
        elif cur_key == "A+X":
            self.get_ready_state = True
            self.init_count = 0
            self.node.get_logger().info(colored("Setting to init state", "blue"))
        elif cur_key == "B+up" and not self.stand_command:
            self.base_height_command[0, 0] += 0.05
            self.node.get_logger().info(colored(f"Base height command: {self.base_height_command[0, 0]}", "green"))
        elif cur_key == "B+down" and not self.stand_command:
            self.base_height_command[0, 0] -= 0.05
            self.node.get_logger().info(colored(f"Base height command: {self.base_height_command[0, 0]}", "green"))
        elif cur_key == "Y+left":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "Y+right":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+left":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+right":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+Y":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.node.get_logger().info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.node.get_logger().info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "L2":
            self.lin_vel_command[0, :] *= 0.
            self.ang_vel_command[0, :] *= 0.
            self.node.get_logger().info(colored("Velocities set to zero", "blue"))
        elif cur_key == "R2":
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
                self.node.get_logger().info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = 0.78
                self.node.get_logger().info(colored("Walk command", "blue"))

    def odometry_callback(self, msg):
        # Extract current position from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.mocap_pos = np.array([x, y, z])
        
        # Convert orientation from quaternion to Euler angles
        quat = msg.pose.pose.orientation
        self.mocap_quat = np.array([quat.x, quat.y, quat.z, quat.w])
        rot = Rotation.from_quat(self.mocap_quat)
        self.mocap_euler = rot.as_euler('xyz')
        twist = msg.twist.twist
        lin_vel = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        lin_vel = rot.inv().apply(lin_vel)
        self.mocap_lin_vel = lin_vel
        # print(f"Current position: {self.mocap_pos}, Current orientation: {self.mocap_euler}")

    def run(self):
        total_inference_cnt = 0
        start_time = time.time()
        try:
            while rclpy.ok():
                if self.use_joystick and self.wc_msg is not None:
                    self.process_joystick_input()
                self.rl_inference()
                end_time = time.time()
                total_inference_cnt += 1

                self.rate.sleep()
        except KeyboardInterrupt:
            pass