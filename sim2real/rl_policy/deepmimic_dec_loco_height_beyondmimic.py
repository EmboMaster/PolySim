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
def quat_to_rotation_matrix_6d(quat):
    """
    Convert quaternion to 6D rotation representation (first 2 columns of rotation matrix).
    
    Args:
        quat: (N, 4) array of quaternions in [w, x, y, z] or [x, y, z, w] format
              Check your data format!
    
    Returns:
        (N, 6) array of 6D rotation representation
    """
    # Assuming quat is [x, y, z, w] format (check your robot_state_data format!)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # If your quat is [w, x, y, z], use:
    # w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Build rotation matrix (3x3)
    # First column
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)
    
    # Second column
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)
    
    # Return first two columns flattened: [r00, r10, r20, r01, r11, r21]
    return np.stack([r00, r10, r20, r01, r11, r21], axis=1)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define wrist joint indices
        # IMPORTANT: Adjust these based on your robot's joint ordering!
        # Example for a typical humanoid:
        self.wrist_joint_names = ["left_wrist_roll", "right_wrist_roll"]  # Adjust names!
        
        # Find wrist joint indices in dof_pos
        # This assumes you have access to joint names somewhere
        # If not, you need to hardcode the indices
        self.wrist_joint_ids = self._find_wrist_joint_indices()
        
        # Create mask for non-wrist joints
        self.non_wrist_joint_mask = np.ones(self.num_dofs, dtype=bool)
        if len(self.wrist_joint_ids) > 0:
            self.non_wrist_joint_mask[self.wrist_joint_ids] = False
        
        print(f"Wrist joint IDs: {self.wrist_joint_ids}")
        print(f"Number of non-wrist DOFs: {np.sum(self.non_wrist_joint_mask)}")

    def _find_wrist_joint_indices(self):
        """Find indices of wrist joints"""
        wrist_ids = []
        
        # Method 1: If you have joint names list
        if hasattr(self, 'joint_names'):
            for i, name in enumerate(self.joint_names):
                if 'wrist' in name.lower():
                    wrist_ids.append(i)
        
        # Method 2: Hardcode based on your robot (ADJUST THIS!)
        else:
            # Example: For H1 robot, wrist joints might be at indices [?, ?]
            # Check your robot's URDF/MJCF to find the correct indices
            wrist_ids = []  # Empty for now, will need to be filled
            
            # Uncomment and adjust based on your robot:
            # wrist_ids = [19, 20]  # Example indices
        
        return wrist_ids
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
        """
        Simplified version for ONLY motion tracking policy.
        Matches the training actor observations exactly.
        """
        # Extract state components
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
        
        # Relative joint positions
        dof_pos_minus_default = dof_pos - self.default_dof_angles
        
        # Projected gravity
        v = np.array([[0, 0, -1]])
        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        # Convert base quaternion to 6D rotation
        robot_anchor_ori_w = quat_to_rotation_matrix_6d(base_quat)
        
        # Filter out wrist joints
        joint_pos_rel_wo_wrist = dof_pos_minus_default[:, self.non_wrist_joint_mask]
        joint_vel_rel_wo_wrist = dof_vel[:, self.non_wrist_joint_mask]
        last_action_wo_wrist = self.last_action[:, self.non_wrist_joint_mask]
        
        # Get motion phase
        self.get_frame_encoding()  # Updates self.phase
        
        # Build observation matching training
        if self.use_history_mimic:
            history_mimic = self._get_obs_history_mimic(self.obs_mimic_dims)
            history_mimic *= self.obs_scales["history_mimic"]
            
            obs = np.concatenate([
                robot_anchor_ori_w,           # 6D base orientation
                base_ang_vel*0.25,            # Base angular velocity
                projected_gravity,            # Projected gravity
                joint_pos_rel_wo_wrist,       # Joint positions (no wrist)
                joint_vel_rel_wo_wrist*0.05,  # Joint velocities (no wrist)
                last_action_wo_wrist,         # Actions (no wrist)
                np.array([[self.phase]]),     # Motion phase
                history_mimic                 # History
            ], axis=1)
        else:
            obs = np.concatenate([
                robot_anchor_ori_w,           # 6D base orientation
                base_ang_vel*0.25,            # Base angular velocity
                projected_gravity,            # Projected gravity
                joint_pos_rel_wo_wrist,       # Joint positions (no wrist)
                joint_vel_rel_wo_wrist*0.05,  # Joint velocities (no wrist)
                last_action_wo_wrist,         # Actions (no wrist)
                np.array([[self.phase]])      # Motion phase
            ], axis=1)
        
        # Update history
        if self.history_handler:
            self.history_handler.add("robot_anchor_ori_w", robot_anchor_ori_w)
            self.history_handler.add("base_ang_vel", base_ang_vel*0.25)
            self.history_handler.add("projected_gravity", projected_gravity)
            self.history_handler.add("joint_pos_rel_wo_wrist", joint_pos_rel_wo_wrist)
            self.history_handler.add("joint_vel_rel_wo_wrist", joint_vel_rel_wo_wrist*0.05)
            self.history_handler.add("actions", last_action_wo_wrist)
            self.history_handler.add("ref_motion_phase", np.array([[self.phase]]))
        
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