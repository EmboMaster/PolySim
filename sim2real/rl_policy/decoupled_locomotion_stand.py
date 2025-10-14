import rclpy
from rclpy.node import Node
import numpy as np
import time
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import threading
# from pynput import keyboard
import argparse
import yaml
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_policy')

from base_policy import BasePolicy
from sshkeyboard import listen_keyboard
from termcolor import colored

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c

def clock_input():
    t = time.time()
    t -= int(t / 1000) * 1000

    frequency = 1.5
    phase = 0.5

    gait_indices = t * frequency - int(t * frequency)
    foot_indices = np.array([phase, 0]) + gait_indices

    clock_inputs = np.sin(2 * np.pi * foot_indices)

    return clock_inputs[None, :]

class DecoupledLocomotionStandPolicy(BasePolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        self.use_clock_input = False

        self.num_upper_dofs = self.config['NUM_UPPER_BODY_JOINTS']
        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        # self.ref_upper_dof_pos[:, 4] = 0.5
        # self.ref_upper_dof_pos[:, 11] = -0.2
        # self.ref_upper_dof_pos[:, 6] = 0.9
        # self.ref_upper_dof_pos[:, 13] = 1.25
        # self.ref_upper_dof_pos[:, 4] = 0.3
        # self.ref_upper_dof_pos[:, 11] = -0.3
        # self.ref_upper_dof_pos[:, 6] = 1.
        # self.ref_upper_dof_pos[:, 13] = 1.
        self.last_policy_action = np.zeros((1, self.num_dofs - self.num_upper_dofs))
        self.gait_period = self.config["GAIT_PERIOD"]
        self.phase_time = np.zeros((1, 1))

    def _get_obs_phase_time(self):
        cur_time = self.node.get_clock().now().nanoseconds / 1e9 * self.stand_command[0, 0]
        # print("cur_time: ", cur_time)
        phase_time = cur_time % self.gait_period / self.gait_period
        # print("phase_time: ", phase_time)
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state [:2]: timestamps
        # robot_state [2:5]: robot base pos
        # robot_state [5:9]: robot base orientation
        # robot_state [9:9+dof_num]: joint angles 
        # robot_state [9+dof_num: 9+dof_num+3]: base linear velocity
        # robot_state [9+dof_num+3: 9+dof_num+6]: base angular velocity
        # robot_state [9+dof_num+6: 9+dof_num+6+dof_num]: joint velocities
        # RL observation preparation
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]


        dof_pos_minus_default = dof_pos - self.default_dof_angles

        v = np.array([[0, 0, -1]])

        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2*np.pi*phase_time)
        cos_phase = np.cos(2*np.pi*phase_time)
        
        # import ipdb; ipdb.set_trace()   
        # print(base_ang_vel)
        if self.use_history:
            history = self._get_obs_history()
            history *= self.obs_scales["history"]
            obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.lin_vel_command,
                                    self.stand_command,
                                    cos_phase,
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    history,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    sin_phase
                                    ], axis=1)
        else:
            obs = np.concatenate([self.last_policy_action, 
                                base_ang_vel*0.25, 
                                self.ang_vel_command, 
                                self.lin_vel_command, 
                                self.stand_command,
                                cos_phase,
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                sin_phase
                                ], axis=1)
        if self.history_handler:
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales["base_ang_vel"])
            self.history_handler.add("command_lin_vel", self.lin_vel_command*self.obs_scales["command_lin_vel"])
            self.history_handler.add("command_ang_vel", self.ang_vel_command*self.obs_scales["command_ang_vel"])
            self.history_handler.add("command_stand", self.stand_command*self.obs_scales["command_stand"])
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales["dof_pos"])
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales["dof_vel"])
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales["projected_gravity"])
            self.history_handler.add("ref_upper_dof_pos", self.ref_upper_dof_pos*self.obs_scales["ref_upper_dof_pos"])
            self.history_handler.add("actions", self.last_policy_action*self.obs_scales["actions"])
            # self.history_handler.add("phase_time", phase_time*self.obs_scales["phase_time"])
            self.history_handler.add("sin_phase", sin_phase*self.obs_scales["sin_phase"])
            self.history_handler.add("cos_phase", cos_phase*self.obs_scales["cos_phase"])
        # examine obs
        # print("last_policy_action", self.last_policy_action)
        # print("base_ang_vel", base_ang_vel)
        # print("ang_vel_command", self.ang_vel_command)
        # print("lin_vel_command", self.lin_vel_command)
        # print("stand_command", self.stand_command)
        # print("dof_pos_minus_default", dof_pos_minus_default)
        # print("dof_vel", dof_vel)
        # print("projected_gravity", projected_gravity)
        # print("ref_upper_dof_pos", self.ref_upper_dof_pos)
        return obs.astype(np.float32)

    def get_policy_action(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # Lower body actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale
        # Combine upper body actions
        scaled_policy_action = np.concatenate([scaled_policy_action, self.ref_upper_dof_pos], axis=1)

        return scaled_policy_action
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    
    rate = node.create_rate(50)


    

    locomotion_policy = DecoupledLocomotionStandPolicy(config=config, 
                                                        node=node, 
                                                        model_path=args.model_path, 
                                                        use_jit=args.use_jit,
                                                        rl_rate=50, 
                                                        decimation=4)

    locomotion_policy.run()