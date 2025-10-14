import rclpy
from rclpy.node import Node
import numpy as np
import time
import pygame
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray, Bool
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import threading
# from pynput import keyboard
from sshkeyboard import listen_keyboard
import argparse
import yaml
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('./rl_policy')

import onnxruntime
# import torch
import os
from loguru import logger

from deepmimic_dec_loco import MotionTrackingDecLocoPolicy

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

class MotionTrackingDecLocoHeightPolicy(MotionTrackingDecLocoPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 loco_model_path, 
                 mimic_model_paths,
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         node, 
                         loco_model_path,
                         mimic_model_paths, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation,
                         use_mocap)
        self.base_height_command = np.array([[0.78]])
        self.disable_elastic_band_pub = self.node.create_publisher(Bool, '/disable_elastic_band', 10)
        self.elastic_band_disabled = False

    def disable_elastic_band(self):
        if not self.elastic_band_disabled:
            msg = Bool()
            msg.data = True
            self.disable_elastic_band_pub.publish(msg)
            self.elastic_band_disabled = True
            logger.info("Sent disable elastic band signal")

    def _get_obs_history_loco_height(self, obs_dims={}):
        assert "history_loco_height_config" in self.config.keys()
        history_config = self.config["history_loco_height_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            obs_dim = obs_dims.get(key, history_array.shape[2])
            history_array = history_array[:, :, :obs_dim] # Shape: [4096, history_length, obs_dim]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)

    def get_frame_encoding(self):
        # 11 bins for 11 seconds, if (current_time-self.frame_start_time) > 1, increment frame_idx
        # the frame encoding is maped to 0-1
        current_time = self.node.get_clock().now().nanoseconds / 1e9
        # import ipdb; ipdb.set_trace()
        motion_length_s = self.motion_length_s[self.policy_mimic_idx]
        self.phase = (current_time - self.frame_start_time) / motion_length_s
        # print("phase", self.phase)
        self.vis_process("Mimic", self.phase)
        if self.phase >= 1.0:
            self.frame_start_time = current_time
            self.phase = 0.0
            # If current mimic policy is done, switch to locomotion policy
            self.policy_locomotion_mimic_flag = 0
            self.policy = self.policy_locomotion
            logger.info(f"\rSwitched to Locomotion policy")
            self.base_height_command = np.array([[0.78]])
            self.end_upper_dof_pos = self.robot_state_data[:, (7+self.num_lower_dofs):(7+self.num_dofs)].copy()
            # zero out the waist roll and pitch
            self.end_upper_dof_pos[:, 1] = 0.0
            self.end_upper_dof_pos[:, 2] = 0.0
            self.ref_upper_dof_pos[0, :] = self.end_upper_dof_pos[0, :].copy()

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
        
        # prepare frame encoding for deepmimic
        if self.policy_locomotion_mimic_flag and self.interpolation_done: self.get_frame_encoding()

        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2*np.pi*phase_time)
        cos_phase = np.cos(2*np.pi*phase_time)

        if not self.policy_locomotion_mimic_flag or not self.interpolation_done:
            if self.use_history_loco:
                history_loco = self._get_obs_history_loco_height(self.obs_loco_dims)
                history_loco *= self.obs_scales["history_loco"]
                obs = np.concatenate([self.last_action[:, :self.num_lower_dofs], 
                                        base_ang_vel*0.25, 
                                        self.ang_vel_command, 
                                        self.base_height_command*2.0,
                                        self.lin_vel_command,
                                        self.stand_command,
                                        cos_phase,
                                        dof_pos_minus_default, 
                                        dof_vel*0.05,
                                        history_loco,
                                        # phase_time,
                                        projected_gravity,
                                        self.ref_upper_dof_pos,
                                        sin_phase
                                        ], axis=1)
            else:
                obs = np.concatenate([self.last_action[:, :self.num_lower_dofs], 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.base_height_command*2.0,
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
        else:
            if self.use_history_mimic:
                history_mimic = self._get_obs_history_mimic(self.obs_mimic_dims)
                history_mimic *= self.obs_scales["history_mimic"]
                obs = np.concatenate([self.last_action[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]],
                                    base_ang_vel*0.25,
                                    dof_pos_minus_default[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]], 
                                    dof_vel[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]]*0.05,
                                    history_mimic,
                                    projected_gravity,
                                    np.array([[self.phase]])
                                    ], axis=1)
            else:
                obs = np.concatenate([self.last_action[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]], 
                                        base_ang_vel*0.25, 
                                        dof_pos_minus_default[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]], 
                                        dof_vel[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]]*0.05,
                                        projected_gravity,
                                        np.array([[self.phase]])
                                        ], axis=1)
        # Yuanhang: update history handler afterwards
        if self.history_handler:
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales["base_ang_vel"])
            self.history_handler.add("command_lin_vel", self.lin_vel_command*self.obs_scales["command_lin_vel"])
            self.history_handler.add("command_ang_vel", self.ang_vel_command*self.obs_scales["command_ang_vel"])
            self.history_handler.add("command_stand", self.stand_command*self.obs_scales["command_stand"])
            self.history_handler.add("command_base_height", self.base_height_command*self.obs_scales["command_base_height"])
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales["dof_pos"])
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales["dof_vel"])
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales["projected_gravity"])
            self.history_handler.add("ref_upper_dof_pos", self.ref_upper_dof_pos*self.obs_scales["ref_upper_dof_pos"])
            self.history_handler.add("actions", self.last_action*self.obs_scales["actions"])
            # self.history_handler.add("phase_time", phase_time*self.obs_scales["phase_time"])
            self.history_handler.add("ref_motion_phase", np.array([[self.phase]])*self.obs_scales["ref_motion_phase"])
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--loco_model_path', type=str, default=None, help='loco model path')
    parser.add_argument('--mimic_model_paths', type=str, default=None, help='mimic model paths')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    rclpy.init(args=None)
    node = rclpy.create_node('simple_node')
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start() 

    locomotion_policy = MotionTrackingDecLocoHeightPolicy(config=config, 
                                                        node=node, 
                                                        loco_model_path=args.loco_model_path, 
                                                        mimic_model_paths=args.mimic_model_paths,
                                                        use_jit=args.use_jit,
                                                        rl_rate=50, 
                                                        decimation=4)
    locomotion_policy.run()
    rclpy.shutdown()