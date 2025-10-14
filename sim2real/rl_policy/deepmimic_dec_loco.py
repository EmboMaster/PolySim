import rclpy
from rclpy.node import Node
import numpy as np
import time
import pygame
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import threading
# from pynput import keyboard
from sshkeyboard import listen_keyboard
import argparse
import yaml
from termcolor import colored
import sys
sys.path.append('./rl_policy')

import onnxruntime
# import torch
from base_policy import BasePolicy
import os
from loguru import logger

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

class MotionTrackingDecLocoPolicy(BasePolicy):
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
        self.mimic_model_paths = mimic_model_paths
        self.policy_locomotion = None
        self.policies_mimic = []
        self.policy_mimic_idx = 0
        self.policy_mimic_names = []
        self.policy_locomotion_mimic_flag = 0 # 0: locomotion, 1: mimic
        self.start_upper_dof_pos = []
        self.end_upper_dof_pos = np.zeros((1, 17))
        self.motion_length_s = []
        self.use_history_loco = config.get("USE_HISTORY_LOCO", False)
        self.use_history_mimic = config.get("USE_HISTORY_MIMIC", False)
        self.obs_loco_dims = config.get("obs_loco_dims", {})
        self.obs_mimic_dims = config.get("obs_mimic_dims", {})
        self.mimic_robot_types = config.get("mimic_robot_types", {})
        self.robot_dofs = config.get("robot_dofs", {})
        self.policy_mimic_robot_types = []
        self.policy_mimic_robot_dofs = []
        # Interpolation variables
        self.interpolation_done = False
        self.interpolation_active = False
        self.interpolation_emergency = False
        self.interpolation_progress = 0.0
        self.interpolation_duration_loco2mimic = 1.5  # interpolation in progress duration in seconds
        self.interpolation_duration_mimic2loco = 1.5  # interpolation in progress duration in seconds
        self.interpolation_duration_loco2mimic_gap = 1.0 # interpolation end duration in seconds
        self.interpolation_duration_mimic2loco_gap = 0.5 # interpolation end duration in seconds
        self.interpolation_threshold_loco2mimic = (self.interpolation_duration_loco2mimic + self.interpolation_duration_loco2mimic_gap) / self.interpolation_duration_loco2mimic
        self.interpolation_threshold_mimic2loco = (self.interpolation_duration_mimic2loco + self.interpolation_duration_mimic2loco_gap) / self.interpolation_duration_mimic2loco
        super().__init__(config, 
                         node, 
                         loco_model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        self.use_clock_input = False
    
        self.robot_state_data = None
        self.last_action = np.zeros((1, self.num_dofs))
        self.num_upper_dofs = self.config['NUM_UPPER_BODY_JOINTS']
        self.num_lower_dofs = self.num_dofs - self.num_upper_dofs
        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos[:, 4] = 0.3
        self.ref_upper_dof_pos[:, 11] = -0.3
        self.ref_upper_dof_pos[:, 6] = 1.
        self.ref_upper_dof_pos[:, 13] = 1.
        self.gait_period = self.config["GAIT_PERIOD"]
        self.phase_time = np.zeros((1, 1))

        self.frame_idx = 0
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        self.frame_start_time = start_time
        self.frame_last_time = start_time
        self.phase = 0.0

    def setup_mimic_policies(self):
        """
        Setup multiple policies and store them in `self.policies_mimic`.

        Args:
            use_jit (bool): Boolean indicating whether to use JIT for all policies.
        """
        # Load mimic models configuration from self.config
        mimic_models_config = self.config.get("mimic_models", {})
        assert mimic_models_config, "No mimic models found in the configuration!"

        for policy_name, relative_path in mimic_models_config.items():
            # Construct full path to the ONNX model
            model_path = os.path.join(self.mimic_model_paths, os.path.join(policy_name, relative_path))
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print(f"Loading mimic policy '{policy_name}' from {model_path}")

            # Load ONNX policy
            onnx_policy_session = onnxruntime.InferenceSession(model_path)
            onnx_input_name = onnx_policy_session.get_inputs()[0].name
            onnx_output_name = onnx_policy_session.get_outputs()[0].name

            # Define the policy function
            def policy_act(obs, session=onnx_policy_session, input_name=onnx_input_name, output_name=onnx_output_name):
                return session.run([output_name], {input_name: obs})[0]

            # Append the policy function and related metadata
            self.policies_mimic.append(policy_act)
            self.policy_mimic_names.append(policy_name)

            # Verify that configuration contains the required metadata for the policy
            assert policy_name in self.config["start_upper_body_dof_pos"], \
                f"Start upper body DOF position not found in config for policy '{policy_name}'"
            self.start_upper_dof_pos.append(np.array(self.config["start_upper_body_dof_pos"][policy_name]))

            assert policy_name in self.config["motion_length_s"], \
                f"Motion length not found in config for policy '{policy_name}'"
            self.motion_length_s.append(self.config["motion_length_s"][policy_name])
            
            self.policy_mimic_robot_types.append(self.mimic_robot_types[policy_name])
            self.policy_mimic_robot_dofs.append(np.array(self.robot_dofs[self.mimic_robot_types[policy_name]], 
                                                         dtype=bool))

        # Record the number of mimic policies loaded
        self.num_mimic_policies = len(self.policies_mimic)
        print(f"Successfully loaded {self.num_mimic_policies} mimic policies.")


    def setup_policy(self, model_path, use_jit):
        """
        Setup all policies here.
        """
        # load onnx policy
        if not use_jit:
            self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
            def policy_act(obs):
                return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        else:
            raise NotImplementedError("JIT not implemented yet.")
        self.policy_locomotion = policy_act
        self.setup_mimic_policies()
        # Default policy is locomotion
        self.policy = self.policy_locomotion
    
    def _get_obs_history_loco(self, obs_dims={}):
        assert "history_loco_config" in self.config.keys()
        history_config = self.config["history_loco_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            obs_dim = obs_dims.get(key, history_array.shape[2])
            history_array = history_array[:, :, :obs_dim] # Shape: [4096, history_length, obs_dim]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)
    
    def _get_obs_history_mimic(self, obs_dims={}):
        assert "history_mimic_config" in self.config.keys()
        history_config = self.config["history_mimic_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            # Get the obs_dim from obs_dims if it exists, otherwise use the full obs_dim
            obs_dim = obs_dims.get(key, history_array.shape[2])
            history_array = history_array[:, :, :obs_dim] # Shape: [4096, history_length, obs_dim]
            # Get the disired history obs elements
            if key == "actions" or key == "dof_pos" or key == "dof_vel":
                history_array = history_array[:, :, self.policy_mimic_robot_dofs[self.policy_mimic_idx]]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)
    
    def next_mimic_policy(self,):
        self.policy_mimic_idx = (self.policy_mimic_idx + 1) % len(self.policies_mimic)
    
    def last_mimic_policy(self,):
        self.policy_mimic_idx = (self.policy_mimic_idx - 1) % len(self.policies_mimic)

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
            self.end_upper_dof_pos = self.robot_state_data[:, (7+self.num_lower_dofs):(7+self.num_dofs)].copy()
            # zero out the waist roll and pitch
            self.end_upper_dof_pos[:, 1] = 0.0
            self.end_upper_dof_pos[:, 2] = 0.0
            self.ref_upper_dof_pos[0, :] = self.end_upper_dof_pos[0, :].copy()

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
        
        # prepare frame encoding for deepmimic
        if self.policy_locomotion_mimic_flag and self.interpolation_done: self.get_frame_encoding()

        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2*np.pi*phase_time)
        cos_phase = np.cos(2*np.pi*phase_time)

        if not self.policy_locomotion_mimic_flag or not self.interpolation_done:
            if self.use_history_loco:
                history_loco = self._get_obs_history_loco(self.obs_loco_dims)
                history_loco *= self.obs_scales["history_loco"]
                obs = np.concatenate([self.last_action[:, :self.num_lower_dofs], 
                                        base_ang_vel*0.25, 
                                        self.ang_vel_command, 
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
    
    def get_policy_action(self, robot_state_data):
        if self.policy_locomotion_mimic_flag:
            # Phase 1: Start interpolation
            if not self.interpolation_done and not self.interpolation_active:
                self.interpolation_progress = 0.0
                self.interpolation_active = True
                self.interpolation_start_pos = self.ref_upper_dof_pos[0, :].copy()
                self.interpolation_target_pos = self.start_upper_dof_pos[self.policy_mimic_idx]
                self.interpolation_target_pos[1:3] *= 0.0
                self.interpolation_start_time = self.node.get_clock().now().nanoseconds / 1e9
                print(f"Interpolating from {self.interpolation_start_pos} to {self.interpolation_target_pos}")
            # Phase 2: Interpolation is in progress
            if not self.interpolation_done and self.interpolation_progress < self.interpolation_threshold_loco2mimic:
                timestep = self.node.get_clock().now().nanoseconds / 1e9 - self.interpolation_start_time
                self.interpolation_progress = timestep / self.interpolation_duration_loco2mimic
                if self.interpolation_progress >= self.interpolation_threshold_loco2mimic:
                    self.interpolation_progress = 1.0
                    self.ref_upper_dof_pos[0, :] = self.interpolation_target_pos.copy()
                    self.interpolation_done = True
                    self.interpolation_active = False
                    # Switch to mimic policy
                    self.policy = self.policies_mimic[self.policy_mimic_idx]
                    self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
                    logger.info(f"\rSwitched to Mimic policy: {self.policy_mimic_names[self.policy_mimic_idx]}")
                    self.history_handler.reset([0])
                    # import ipdb; ipdb.set_trace()
                else:
                    alpha = min(self.interpolation_progress, 1.0)
                    self.ref_upper_dof_pos[0, :] = (1 - alpha) * self.interpolation_start_pos + alpha * self.interpolation_target_pos
                    self.vis_process("Interpolation", alpha)
            # Phase 3: Interpolation is done, use mimic policy
            if self.interpolation_done:
                obs = self.prepare_obs_for_rl(robot_state_data)
                if self.policy_locomotion_mimic_flag:
                    policy_action = np.clip(self.policy(obs), -100, 100)
                    full_policy_action = np.zeros((1, self.num_dofs))
                    full_policy_action[:, self.policy_mimic_robot_dofs[self.policy_mimic_idx]] = policy_action
                    self.last_action = full_policy_action.copy()
                    scaled_policy_action = full_policy_action * self.policy_action_scale
                else:
                    # Go to Phase 4, doing nothing here
                    pass
        # Phase 4: Mimic policy is done or emergency stop, switch back to locomotion policy
        if not self.policy_locomotion_mimic_flag or not self.interpolation_done:
            self.policy = self.policy_locomotion
            obs = self.prepare_obs_for_rl(robot_state_data)
            policy_action = np.clip(self.policy(obs), -100, 100)
            self.last_action[:, :self.num_lower_dofs] = policy_action.copy()
            self.last_action[:, self.num_lower_dofs:] = self.ref_upper_dof_pos.copy()
            # Lower body actions
            scaled_policy_action = policy_action * self.policy_action_scale
            # Interpolate back to default upper body dof pos for locomotion
            if self.interpolation_done or self.interpolation_emergency:
                # Update reference upper dof pos
                timestep = self.node.get_clock().now().nanoseconds / 1e9 - self.frame_start_time
                alpha = max((timestep - self.interpolation_duration_mimic2loco_gap)/self.interpolation_duration_mimic2loco, 0.0)
                self.ref_upper_dof_pos = (1 - alpha) * self.end_upper_dof_pos + alpha * np.array(self.config["loco_upper_body_dof_pos"])
                self.vis_process("Interpolation", alpha)
                if alpha >= 1.0:
                    self.interpolation_done = False
                    self.interpolation_emergency = False
                    self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            # Combine upper body actions
            scaled_policy_action = np.concatenate([scaled_policy_action, self.ref_upper_dof_pos], axis=1)

        return scaled_policy_action
        
    def vis_process(self, name, alpha):
        bar_length = 30  # Adjust bar length as needed
        filled_length = int(bar_length * alpha)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\r{name} Progress: |{bar}| {alpha:.2%}")
        sys.stdout.flush()
        
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "[":
            self.disable_elastic_band()
            self.policy_locomotion_mimic_flag = 1 - self.policy_locomotion_mimic_flag
            self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            if self.policy_locomotion_mimic_flag == 1:
                # reset interpolation variables
                self.interpolation_done = False
            else: 
                self.interpolation_emergency = True
                self.end_upper_dof_pos = self.robot_state_data[:, (7+self.num_lower_dofs):(7+self.num_dofs)].copy()
                # zero out the waist roll and pitch
                # self.end_upper_dof_pos[:, 1] = 0.0
                # self.end_upper_dof_pos[:, 2] = 0.0
                self.ref_upper_dof_pos[0, :] = self.end_upper_dof_pos[0, :].copy()
        elif keycode == ";":
            # only switch to next policy if current policy is locomotion
            if self.policy_locomotion_mimic_flag == 0:
                self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
                self.next_mimic_policy()
        elif keycode == "'":
            # only switch to last policy if current policy is locomotion
            if self.policy_locomotion_mimic_flag == 0:
                self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
                self.last_mimic_policy()
        if self.policy_locomotion_mimic_flag == 0:
            print(f"Current policy: locomotion")
        else:
            print(f"Current policy: mimic[{self.policy_mimic_idx}]")
            print(f"Current checkpoint path: {self.mimic_model_paths[self.policy_mimic_idx]}")
        print(f"Current mimic policy name: {self.policy_mimic_names[self.policy_mimic_idx]}")
        print(f"Current motion length: {self.motion_length_s[self.policy_mimic_idx]}")
        # print checkpint path
        print(f"Current checkpoint path: {self.mimic_model_paths[self.policy_mimic_idx]}")
        

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "select":
            self.history_handler.reset([0])
            self.policy_locomotion_mimic_flag = 1 - self.policy_locomotion_mimic_flag
            self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            if self.policy_locomotion_mimic_flag == 1:
                # Reset interpolation variables
                self.interpolation_done = False
            else: 
                self.interpolation_emergency = True
                self.end_upper_dof_pos = self.robot_state_data[:, (7+self.num_lower_dofs):(7+self.num_dofs)].copy()
                self.ref_upper_dof_pos[0, :] = self.end_upper_dof_pos[0, :].copy()
                self.node.get_logger().info(colored("Current policy: Locomotion", "blue"))
        elif cur_key == "R1":
            self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            self.next_mimic_policy()
            self.node.get_logger().info(colored(f"Current Mimic: {self.policy_mimic_names[self.policy_mimic_idx]}, length: {self.motion_length_s[self.policy_mimic_idx]}", "blue"))
            # path
            self.node.get_logger().info(colored(f"Current checkpoint path: {self.mimic_model_paths}", "blue"))
        elif cur_key == "L1":
            self.frame_start_time = self.node.get_clock().now().nanoseconds / 1e9
            self.last_mimic_policy()
            self.node.get_logger().info(colored(f"Current Mimic: {self.policy_mimic_names[self.policy_mimic_idx]}, length: {self.motion_length_s[self.policy_mimic_idx]}", "blue"))
            # path
            self.node.get_logger().info(colored(f"Current checkpoint path: {self.mimic_model_paths}", "blue"))
    
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

    locomotion_policy = MotionTrackingDecLocoPolicy(config=config, 
                                                    node=node, 
                                                    loco_model_path=args.loco_model_path, 
                                                    mimic_model_paths=args.mimic_model_paths,
                                                    use_jit=args.use_jit,
                                                    rl_rate=50, 
                                                    decimation=4)
    locomotion_policy.run()
    rclpy.shutdown()