import torch
import math
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib

class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        
        super().__init__(config, device)
        self.use_adaptive_sampling = False
        self._init_motion_lib()
        self._init_adaptive_sampling()
        self._init_motion_extend()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True

        self._init_save_motion()

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")



        

    def teleop_callback(self, msg):
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        if len(env_ids) > 0:
            self._debug_reset_envs = env_ids.clone()

    def _init_save_motion(self):
        if "save_motion" in self.config:
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)

                
                if hasattr(self.config, 'dump_motion_name'):
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
                self.motion_times_buf = []
                self.start_save = False

        else:
            self.save_motion = False

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)
            
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
    
    def _init_adaptive_sampling(self):
        motion_cfg = self.config.robot.motion
        self.use_adaptive_sampling = getattr(motion_cfg, "adaptive_sampling", False)
        if not self.use_adaptive_sampling:
            return
        max_motion_len = self._motion_lib.get_motion_length().max()
        self.bin_count = int((max_motion_len / self.dt).item()) + 1
        self.adaptive_uniform_ratio = getattr(motion_cfg, "adaptive_uniform_ratio", 0.1)
        self.adaptive_alpha = getattr(motion_cfg, "adaptive_alpha", 0.1)
        self.adaptive_kernel_size = getattr(motion_cfg, "adaptive_kernel_size", 5)
        self.adaptive_lambda = getattr(motion_cfg, "adaptive_lambda", 0.8)

        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
        kernel = torch.tensor([self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)], device=self.device)
        self._adaptive_kernel = kernel / kernel.sum()

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if hasattr(self.config.robot, "anchor_body_name"):
            self.anchor_body_id = self.simulator._body_list.index(self.config.robot.anchor_body_name)
        if hasattr(self.config.robot, "termination_body_names"):
            self.motion_body_pos_z_only_body_ids = [
                self.simulator._body_list.index(name) for name in self.config.robot.termination_body_names
            ]
        if hasattr(self.config, "termination") and hasattr(self.config.termination, "motion_body_pos_z_only_body_names"):
            self.motion_body_pos_z_only_body_ids = [
                self.simulator._body_list.index(name) for name in self.config.termination.motion_body_pos_z_only_body_names
            ]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        if hasattr(self.config.robot, "right_wrist_body_name"):
            self.right_wrist_body_id = self.simulator._body_list.index(self.config.robot.right_wrist_body_name)
        if hasattr(self.config.robot, "left_wrist_body_name"):
            self.left_wrist_body_id = self.simulator._body_list.index(self.config.robot.left_wrist_body_name)
        
        # Identify wrist joint indices
        self.wrist_joint_ids = []
        for i, joint_name in enumerate(self.simulator.dof_names):
            if "wrist" in joint_name.lower():
                self.wrist_joint_ids.append(i)
        self.wrist_joint_ids = torch.tensor(self.wrist_joint_ids, device=self.device, dtype=torch.long)
        
        # Create mask for non-wrist joints
        self.non_wrist_joint_mask = torch.ones(self.simulator.num_dof, dtype=torch.bool, device=self.device)
        if len(self.wrist_joint_ids) > 0:
            self.non_wrist_joint_mask[self.wrist_joint_ids] = False

    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            #self.extend_body_parent_ids = torch.tensor([19, 23, 15])
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs, 
                                         self.num_bodies + self.num_extend_bodies, 
                                         3, 
                                         dtype=torch.float, 
                                         device=self.device, 
                                         requires_grad=False) # extend
            
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        super()._init_buffers()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.ref_root_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_root_rot_w = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_anchor_rot_w = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.robot_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.robot_anchor_rot_w = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.robot_anchor_lin_vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.robot_anchor_ang_vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        if self.use_adaptive_sampling:
            self._update_sampling_bins(env_ids)
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids) # need to resample before reset root states
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
        

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _check_termination(self):
        super()._check_termination()

        term_reasons = {}

        if self.config.termination.terminate_when_motion_far:
            term_reasons["motion_far"] = torch.any(
                torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold,
                dim=-1,
            )
            self.reset_buf |= term_reasons["motion_far"]
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(
                    self.terminate_when_motion_far_threshold, dtype=torch.float
                )

        if getattr(self.config.termination, "terminate_when_anchor_pos_z_only", False):
            threshold = self.config.termination_scales.termination_anchor_pos_z_threshold
            anchor_z_diff = torch.abs(self.ref_anchor_pos_w[:, -1] - self.robot_anchor_pos_w[:, -1])
            term_reasons["anchor_pos_z_only"] = anchor_z_diff > threshold
            self.reset_buf |= term_reasons["anchor_pos_z_only"]

        if getattr(self.config.termination, "terminate_when_anchor_bad_ori", False):
            threshold = self.config.termination_scales.termination_anchor_ori_threshold
            motion_proj_grav = quat_rotate_inverse(self.ref_anchor_rot_w, self.gravity_vec, w_last=True)
            robot_proj_grav = quat_rotate_inverse(self.robot_anchor_rot_w, self.gravity_vec, w_last=True)
            term_reasons["anchor_bad_ori"] = torch.abs(motion_proj_grav[:, 2] - robot_proj_grav[:, 2]) > threshold
            self.reset_buf |= term_reasons["anchor_bad_ori"]

        if getattr(self.config.termination, "terminate_when_motion_body_pos_z_only", False):
            threshold = self.config.termination_scales.termination_motion_body_pos_z_threshold
            if hasattr(self, "motion_body_pos_z_only_body_ids"):
                body_ids = self.motion_body_pos_z_only_body_ids
            else:
                body_ids = list(range(self.ref_body_pos_extend.shape[1]))
            anchor_pos_w_repeat = self.ref_anchor_pos_w.unsqueeze(1).repeat(1, len(body_ids), 1)
            robot_anchor_pos_w_repeat = self.robot_anchor_pos_w.unsqueeze(1).repeat(1, len(body_ids), 1)

            delta_pos_w = robot_anchor_pos_w_repeat.clone()
            delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
            delta_ori_w = calc_heading_quat(
                quat_mul(
                    self.robot_anchor_rot_w,
                    quat_conjugate(self.ref_anchor_rot_w, w_last=True),
                    w_last=True,
                ),
                w_last=True,
            ).unsqueeze(1).repeat(1, len(body_ids), 1)

            ref_body_pos_relative = delta_pos_w + my_quat_rotate(
                delta_ori_w.reshape(-1, 4),
                (self.ref_body_pos_extend[:, body_ids] - anchor_pos_w_repeat).reshape(-1, 3),
            ).view(-1, len(body_ids), 3)
            error_z = torch.abs(ref_body_pos_relative[..., 2] - self._rigid_body_pos_extend[:, body_ids, 2])
            term_reasons["motion_body_pos_z_only"] = torch.any(error_z > threshold, dim=-1)
            self.reset_buf |= term_reasons["motion_body_pos_z_only"]

        # total_from_motion = int(sum(v.sum().item() for v in term_reasons.values()))
        # if total_from_motion > 0:
        #     reason_counts = {k: int(v.sum().item()) for k, v in term_reasons.items()}
        #     logger.info(
        #         f"Termination reasons (motion) at step {self.common_step_counter}: total={total_from_motion}, {reason_counts}"
        #     )
        

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len

    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            if self.use_adaptive_sampling:
                self._adaptive_sampling(env_ids)
            else:
                self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))

    def _update_sampling_bins(self, env_ids):
        if len(env_ids) == 0:
            return
        failed_env_ids = env_ids[~self.time_out_buf[env_ids]]
        if len(failed_env_ids) == 0:
            return
        current_time = self.episode_length_buf[failed_env_ids] * self.dt + self.motion_start_times[failed_env_ids]
        motion_len = self.motion_len[failed_env_ids].clamp(min=1e-6)
        phase = torch.clip(current_time / motion_len, 0.0, 1.0)
        motion_ids = self.motion_ids[failed_env_ids]
        motion_num_steps = self._motion_lib.get_motion_num_steps(motion_ids).to(self.device)
        time_steps = torch.clamp((phase * (motion_num_steps - 1)).long(), min=0)
        denom = torch.clamp(motion_num_steps, min=1)
        bin_idx = torch.clamp((time_steps * self.bin_count) // denom, 0, self.bin_count - 1)
        self._current_bin_failed.index_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float))

    def _adaptive_sampling(self, env_ids):
        if len(env_ids) == 0:
            return
        sampling_prob = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.bin_count)
        sampling_prob = sampling_prob / sampling_prob.sum()

        pad = (0, self.adaptive_kernel_size - 1)
        sampling_prob = torch.nn.functional.pad(sampling_prob.view(1, 1, -1), pad, mode="replicate")
        sampling_prob = torch.nn.functional.conv1d(sampling_prob, self._adaptive_kernel.view(1, 1, -1)).view(-1)
        sampling_prob = sampling_prob / sampling_prob.sum()

        sampled_bins = torch.multinomial(sampling_prob, len(env_ids), replacement=True)
        rand = torch.rand(len(env_ids), device=self.device)
        phase = (sampled_bins.to(torch.float) + rand) / float(self.bin_count)
        motion_len = self.motion_len[env_ids]
        self.motion_start_times[env_ids] = phase * motion_len

        H = -(sampling_prob * (sampling_prob + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_prob.max(dim=0)
        self.log_dict["sampling_entropy"] = H_norm
        self.log_dict["sampling_top1_prob"] = pmax
        self.log_dict["sampling_top1_bin"] = imax.float() / self.bin_count

        self.bin_failed_count = (
            self.adaptive_alpha * self._current_bin_failed + (1 - self.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()


    def _pre_compute_observations_callback(self):
        """Modified version with new observation computations"""
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        self.ref_root_pos_w[:] = motion_res["root_pos"]
        self.ref_root_rot_w[:] = motion_res["root_rot"]

        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend
        ref_body_vel_extend = motion_res["body_vel_t"]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]
        ref_joint_pos = motion_res["dof_pos"]
        ref_joint_vel = motion_res["dof_vel"]
        # Cache reference joint targets for observation use.
        self.ref_joint_pos = ref_joint_pos
        self.ref_joint_vel = ref_joint_vel

        # =================== EXTEND Rigid body computations (keep existing) ===================
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)
        
        extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
        
        self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, 
                                                    self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
        
        self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, 
                                                    self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

        # =================== NEW: Robot anchor (base) information ===================
        # Using base as anchor from robot_root_states
        # self.robot_anchor_pos_w = self.simulator.robot_root_states[:, :3]  # base position
        # # self.robot_anchor_quat_w = self.simulator.robot_root_states[:, 3:7]  # base quaternion (xyzw)
        # self.robot_anchor_lin_vel_w = self.simulator.robot_root_states[:, 7:10]  # base linear velocity
        # self.robot_anchor_ang_vel_w = self.simulator.robot_root_states[:, 10:13]  # base angular velocity
        
        # For compatibility with existing code that uses anchor_body_id
        self.ref_anchor_pos_w[:] = ref_body_pos_extend[:, self.anchor_body_id]
        self.ref_anchor_rot_w[:] = ref_body_rot_extend[:, self.anchor_body_id]
        self.robot_anchor_pos_w[:] = self._rigid_body_pos_extend[:, self.anchor_body_id]
        self.robot_anchor_rot_w[:] = self._rigid_body_rot_extend[:, self.anchor_body_id]
        self.robot_anchor_lin_vel_w[:] = self._rigid_body_vel_extend[:, self.anchor_body_id]
        self.robot_anchor_ang_vel_w[:] = self._rigid_body_ang_vel_extend[:, self.anchor_body_id]
        # =================== NEW: Reference motion anchor/wrist information ===================
        # Motion anchor is the base in reference motion
        # self.motion_anchor_pos_w = self.ref_root_pos_w
        # self.motion_anchor_quat_w = self.ref_root_rot_w
        
        # Motion wrist positions and orientations from reference motion
        if hasattr(self, "right_wrist_body_id"):
            self.motion_right_wrist_pos_w = ref_body_pos_extend[:, self.right_wrist_body_id]
            self.motion_right_wrist_quat_w = ref_body_rot_extend[:, self.right_wrist_body_id]
        
        if hasattr(self, "left_wrist_body_id"):
            self.motion_left_wrist_pos_w = ref_body_pos_extend[:, self.left_wrist_body_id]
            self.motion_left_wrist_quat_w = ref_body_rot_extend[:, self.left_wrist_body_id]
        
        # =================== NEW: Robot body positions/orientations in anchor frame ===================
        num_bodies = self._rigid_body_pos_extend.shape[1]
        
        # Transform robot body positions to anchor frame
        robot_anchor_pos_repeat = self.robot_anchor_pos_w.unsqueeze(1).repeat(1, num_bodies, 1)
        robot_anchor_quat_repeat = self.robot_anchor_rot_w.unsqueeze(1).repeat(1, num_bodies, 1)
        
        # Position in anchor frame: rotate (body_pos - anchor_pos) by inverse anchor rotation
        anchor_inv_quat = quat_conjugate(robot_anchor_quat_repeat.reshape(-1, 4), w_last=True)
        body_pos_diff = (self._rigid_body_pos_extend - robot_anchor_pos_repeat).reshape(-1, 3)
        self.robot_body_pos_b = my_quat_rotate(anchor_inv_quat, body_pos_diff).view(B, num_bodies, 3)
        
        # Orientation in anchor frame: q_body_in_anchor = q_anchor^-1 * q_body
        self.robot_body_quat_b = quat_mul(
            anchor_inv_quat,
            self._rigid_body_rot_extend.reshape(-1, 4),
            w_last=True
        ).view(B, num_bodies, 4)
        
        # =================== NEW: Motion anchor/wrist in robot anchor frame ===================
        # Motion anchor position in robot anchor frame
        anchor_inv_quat_single = quat_conjugate(self.robot_anchor_rot_w, w_last=True)
        motion_anchor_pos_diff = self.ref_anchor_pos_w - self.robot_anchor_pos_w
        self.motion_anchor_pos_b = my_quat_rotate(anchor_inv_quat_single, motion_anchor_pos_diff)
        
        # Motion anchor orientation in robot anchor frame
        self.motion_anchor_quat_b = quat_mul(anchor_inv_quat_single, self.ref_anchor_rot_w, w_last=True)
        
        # Motion wrist positions and orientations in robot anchor frame
        if hasattr(self, "right_wrist_body_id"):
            motion_right_wrist_pos_diff = self.motion_right_wrist_pos_w - self.robot_anchor_pos_w
            self.motion_right_wrist_pos_b = my_quat_rotate(anchor_inv_quat_single, motion_right_wrist_pos_diff)
            self.motion_right_wrist_quat_b = quat_mul(anchor_inv_quat_single, self.motion_right_wrist_quat_w, w_last=True)
        
        if hasattr(self, "left_wrist_body_id"):
            motion_left_wrist_pos_diff = self.motion_left_wrist_pos_w - self.robot_anchor_pos_w
            self.motion_left_wrist_pos_b = my_quat_rotate(anchor_inv_quat_single, motion_left_wrist_pos_diff)
            self.motion_left_wrist_quat_b = quat_mul(anchor_inv_quat_single, self.motion_left_wrist_quat_w, w_last=True)
        
        # =================== Compute differences (keep existing) ===================
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        self.dif_global_body_rot = quat_mul(ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        # if hasattr(self, "_debug_reset_envs") and self._debug_reset_envs.numel() > 0:
        #     reset_envs = self._debug_reset_envs
        #     mask = self.episode_length_buf[reset_envs] == 1
        #     if torch.any(mask):
        #         sel = reset_envs[mask][:8]
        #         print(
        #             f"[Reset AngVel] step={int(self.episode_length_buf[sel[0]].item())} envs={sel.tolist()} "
        #             f"diff={self.dif_global_body_ang_vel[sel]} ref={ref_body_ang_vel_extend[sel]} "
        #             f"robot={self._rigid_body_ang_vel_extend[sel]}"
        #         )
        #         self._debug_reset_envs = torch.zeros(0, device=self.device, dtype=reset_envs.dtype)

        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel

        # Marker coords for visualization (keep existing)
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)

        # =================== Motion phase (keep existing) ===================
        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
        
        self._log_motion_tracking_info()

    def _debug_after_reset_write(self, env_ids):
        """Debug: print reset-time body ang vel diff/ref/robot right after writing states to sim."""
        if self.config.simulator.config.name != "isaacsim":
            return
        if not hasattr(self, "_debug_reset_envs") or self._debug_reset_envs.numel() == 0:
            return
        sel = self._debug_reset_envs
        if sel.numel() == 0:
            return
        # Refresh sim tensors to read back written state.
        try:
            self.simulator.scene.update(dt=0.0)
        except Exception:
            pass
        self.simulator.refresh_sim_tensors()

        # Reference at reset time (episode_length_buf already reset to 0 for these envs).
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=self.env_origins)
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]

        robot_body_ang_vel_extend = torch.cat(
            [self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]],
            dim=1,
        )
        diff = ref_body_ang_vel_extend - robot_body_ang_vel_extend

        sel = sel[:8]
        print(
            f"[ResetWrite AngVel] step={int(self.episode_length_buf[sel[0]].item())} envs={sel.tolist()} "
            f"diff={diff[sel]} ref={ref_body_ang_vel_extend[sel]} robot={robot_body_ang_vel_extend[sel]}"
        )

    def _compute_reward(self):
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                # draw marker joints
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    # import ipdb; ipdb.set_trace()
                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)
                break


            else:
                # draw teleop joints
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # trimesh
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            # self.robot_root_states[env_ids, 2] += 0.04 # in case under the terrain
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'maniskill':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            

        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)


            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]

            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'maniskill':
                # self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
                # root_pose_modified = self.simulator._env.agent.robot.get_root_pose()
                # root_pose_modified.p[env_ids] = root_pos + torch.randn_like(root_pos) * root_pos_noise
                # root_pose_modified.q[env_ids] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                # self.simulator._env.agent.robot.set_root_pose(root_pose_modified)
                # self.simulator.robot_root_states[:, :7] = root_pose_modified.raw_pose
                # # self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                # root_linear_velocity_modified = self.simulator._env.agent.robot.get_root_linear_velocity()
                # root_angular_velocity_modified = self.simulator._env.agent.robot.get_root_angular_velocity()
                # root_linear_velocity_modified[env_ids] = root_vel + torch.randn_like(root_vel) * root_vel_noise
                # root_angular_velocity_modified[env_ids] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
                # self.simulator._env.agent.robot.set_root_linear_velocity(root_linear_velocity_modified)
                # self.simulator._env.agent.robot.set_root_angular_velocity(root_angular_velocity_modified)
                # self.simulator.robot_root_states[:, 7:10] = self.simulator._env.agent.robot.get_root_linear_velocity()
                # self.simulator.robot_root_states[:, 10:13] = self.simulator._env.agent.robot.get_root_angular_velocity()
                # self.simulator._env.scene._gpu_apply_all()  # CPU -> GPU
                # self.simulator._env.scene._gpu_fetch_all()  # GPU -> CPU
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise


    def small_random_quaternions(self, n, max_angle):
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            # Convert angle-axis to quaternion
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
        offset = self.env_origins
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        # import pdb;pdb.set_trace()
        # if self.config.simulator.config['name'] == "maniskill":
        #     qpos_now = self.simulator._env.agent.robot.qpos
        #     qvel_now = self.simulator._env.agent.robot.qvel
        #     qpos_now[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        #     qvel_now[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
        #     self.simulator._env.agent.robot.set_qpos(qpos_now)
        #     self.simulator._env.agent.robot.set_qvel(qvel_now)
        #     self.simulator.dof_pos[env_ids] = qpos_now[env_ids]
        #     self.simulator.dof_vel[env_ids] = qvel_now[env_ids]
        # else:
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise


    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

            if (len(self.motions_for_saving['dof'])) > self.config.save_total_steps:
                for k, v in self.motions_for_saving.items():
                    self.motions_for_saving[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                
                self.motions_for_saving['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                keys_to_save = self.motions_for_saving.keys()

                for i in range(num_motions):
                    motion_key = f"motion{i}" 
                    dump_data[motion_key] = {
                        key: self.motions_for_saving[key][i] for key in keys_to_save
                    }
                    dump_data[motion_key]['fps'] = 1 / self.dt
    
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                
                print(colored(f"Saved motion data to {self.save_motion_dir}.pkl", 'green'))
                import sys
                sys.exit()

            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw   
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "maniskill":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            else:
                raise NotImplementedError
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
            dof = self.simulator.dof_pos.cpu()
            # T, num_env, J, 3
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

    # ############################################################
        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_ref_motion_phase(self):
        return self._ref_motion_phase
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_robot_anchor_ori_w(self):
        """Robot anchor (base) orientation as rotation matrix (first 2 columns flattened, 6D)"""
        # Convert quaternion to rotation matrix
        quat = self.robot_anchor_rot_w  # [B, 4] in xyzw format
        
        # Extract components
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
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
        return torch.stack([r00, r10, r20, r01, r11, r21], dim=1)


    def _get_obs_robot_anchor_lin_vel_w(self):
        """Robot anchor (base) linear velocity in world frame"""
        return self.robot_anchor_lin_vel_w


    def _get_obs_robot_anchor_ang_vel_w(self):
        """Robot anchor (base) angular velocity in world frame"""
        return self.robot_anchor_ang_vel_w


    def _get_obs_base_ang_vel(self):
        """Robot root angular velocity in root (base) frame (body frame)."""
        if hasattr(self, "base_ang_vel"):
            return self.base_ang_vel
        if hasattr(self.simulator, "base_quat") and hasattr(self.simulator, "robot_root_states"):
            return quat_rotate_inverse(self.simulator.base_quat, self.simulator.robot_root_states[:, 10:13], w_last=True)
        # return torch.zeros(self.num_envs, 3, device=self.device)


    def _get_obs_robot_body_pos_b(self):
        """Robot body positions in anchor (base) frame"""
        # Shape: [B, num_bodies, 3] -> [B, num_bodies*3]
        return self.robot_body_pos_b.reshape(self.num_envs, -1)


    def _get_obs_robot_body_ori_b(self):
        """Robot body orientations in anchor (base) frame as rotation matrix (first 2 columns)"""
        # Shape: [B, num_bodies, 4] quaternions
        quat = self.robot_body_quat_b.reshape(-1, 4)  # [B*num_bodies, 4]
        
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # First column
        r00 = 1 - 2*(y*y + z*z)
        r10 = 2*(x*y + w*z)
        r20 = 2*(x*z - w*y)
        
        # Second column
        r01 = 2*(x*y - w*z)
        r11 = 1 - 2*(x*x + z*z)
        r21 = 2*(y*z + w*x)
        
        # Stack and reshape: [B*num_bodies, 6] -> [B, num_bodies*6]
        mat_6d = torch.stack([r00, r10, r20, r01, r11, r21], dim=1)
        return mat_6d.reshape(self.num_envs, -1)


    def _get_obs_motion_anchor_pos_b(self):
        """Motion anchor (reference base) position in robot anchor frame"""
        return self.motion_anchor_pos_b


    def _get_obs_motion_anchor_ori_b(self):
        """Motion anchor (reference base) orientation in robot anchor frame as rotation matrix"""
        quat = self.motion_anchor_quat_b  # [B, 4]
        
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        r00 = 1 - 2*(y*y + z*z)
        r10 = 2*(x*y + w*z)
        r20 = 2*(x*z - w*y)
        r01 = 2*(x*y - w*z)
        r11 = 1 - 2*(x*x + z*z)
        r21 = 2*(y*z + w*x)
        
        return torch.stack([r00, r10, r20, r01, r11, r21], dim=1)


    def _get_obs_motion_right_wrist_pos_b(self):
        """Motion right wrist position in robot anchor frame"""
        if hasattr(self, "motion_right_wrist_pos_b"):
            return self.motion_right_wrist_pos_b
        else:
            return torch.zeros(self.num_envs, 3, device=self.device)


    def _get_obs_motion_left_wrist_pos_b(self):
        """Motion left wrist position in robot anchor frame"""
        if hasattr(self, "motion_left_wrist_pos_b"):
            return self.motion_left_wrist_pos_b
        else:
            return torch.zeros(self.num_envs, 3, device=self.device)


    def _get_obs_motion_right_wrist_ori_b(self):
        """Motion right wrist orientation in robot anchor frame as rotation matrix"""
        if hasattr(self, "motion_right_wrist_quat_b"):
            quat = self.motion_right_wrist_quat_b
            x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            
            r00 = 1 - 2*(y*y + z*z)
            r10 = 2*(x*y + w*z)
            r20 = 2*(x*z - w*y)
            r01 = 2*(x*y - w*z)
            r11 = 1 - 2*(x*x + z*z)
            r21 = 2*(y*z + w*x)
            
            return torch.stack([r00, r10, r20, r01, r11, r21], dim=1)
        else:
            return torch.zeros(self.num_envs, 6, device=self.device)


    def _get_obs_motion_left_wrist_ori_b(self):
        """Motion left wrist orientation in robot anchor frame as rotation matrix"""
        if hasattr(self, "motion_left_wrist_quat_b"):
            quat = self.motion_left_wrist_quat_b
            x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            
            r00 = 1 - 2*(y*y + z*z)
            r10 = 2*(x*y + w*z)
            r20 = 2*(x*z - w*y)
            r01 = 2*(x*y - w*z)
            r11 = 1 - 2*(x*x + z*z)
            r21 = 2*(y*z + w*x)
            
            return torch.stack([r00, r10, r20, r01, r11, r21], dim=1)
        else:
            return torch.zeros(self.num_envs, 6, device=self.device)


    def _get_obs_joint_pos_rel_wo_wrist(self):
        """Joint positions relative to defaults, excluding wrist joints"""
        if hasattr(self, "non_wrist_joint_mask") and len(self.wrist_joint_ids) > 0:
            return self.simulator.dof_pos[:, self.non_wrist_joint_mask]
        else:
            return self.simulator.dof_pos


    def _get_obs_joint_vel_rel_wo_wrist(self):
        """Joint velocities excluding wrist joints"""
        if hasattr(self, "non_wrist_joint_mask") and len(self.wrist_joint_ids) > 0:
            return self.simulator.dof_vel[:, self.non_wrist_joint_mask]
        else:
            return self.simulator.dof_vel

    def _get_obs_generated_commands(self):
        """Reference joint position and velocity from motion (concat)."""
        if hasattr(self, "ref_joint_pos") and hasattr(self, "ref_joint_vel"):
            return torch.cat([self.ref_joint_pos, self.ref_joint_vel], dim=1)
        # Fallback: compute from motion library if cache missing.
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=self.env_origins)
        return torch.cat([motion_res["dof_pos"], motion_res["dof_vel"]], dim=1)

    ###############################################################

    def _reward_motion_global_anchor_position_error_exp(self):
        root_pos_diff = self.ref_anchor_pos_w - self.robot_anchor_pos_w
        error = torch.sum(root_pos_diff**2, dim=-1)
        return torch.exp(-error / self.config.rewards.reward_tracking_sigma.motion_global_anchor_position**2)

    def _reward_motion_global_anchor_orientation_error_exp(self):
        root_rot_diff = quat_mul(
            self.ref_anchor_rot_w,
            quat_conjugate(self.robot_anchor_rot_w, w_last=True),
            w_last=True,
        )
        rotation_diff = quat_to_angle_axis(root_rot_diff)[0]
        # rotation_diff is per-env angle (shape: [num_envs]), so do not reduce to scalar
        error = rotation_diff**2
        return torch.exp(-error / self.config.rewards.reward_tracking_sigma.motion_global_anchor_orientation**2)

    def _reward_motion_relative_body_position_error_exp(self):
        anchor_pos_w_repeat = self.ref_anchor_pos_w.unsqueeze(1).repeat(1, self.ref_body_pos_extend.shape[1], 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w.unsqueeze(1).repeat(
            1, self.ref_body_pos_extend.shape[1], 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = calc_heading_quat(
            quat_mul(
                self.robot_anchor_rot_w,
                quat_conjugate(self.ref_anchor_rot_w, w_last=True),
                w_last=True,
            ),
            w_last=True,
        ).unsqueeze(1).repeat(1, self.ref_body_pos_extend.shape[1], 1)

        ref_body_pos_relative = delta_pos_w + my_quat_rotate(
            delta_ori_w.reshape(-1, 4),
            (self.ref_body_pos_extend - anchor_pos_w_repeat).reshape(-1, 3),
        ).view_as(self.ref_body_pos_extend)

        error = torch.sum((ref_body_pos_relative - self._rigid_body_pos_extend) ** 2, dim=-1)
        return torch.exp(-error.mean(-1) / self.config.rewards.reward_tracking_sigma.motion_relative_body_position**2)

    def _reward_motion_relative_body_orientation_error_exp(self):
        delta_ori_w = calc_heading_quat(
            quat_mul(
                self.robot_anchor_rot_w,
                quat_conjugate(self.ref_anchor_rot_w, w_last=True),
                w_last=True,
            ),
            w_last=True,
        ).unsqueeze(1).repeat(1, self.ref_body_rot_extend.shape[1], 1)

        ref_body_quat_relative = quat_mul(delta_ori_w, self.ref_body_rot_extend, w_last=True)
        rot_diff = quat_mul(
            ref_body_quat_relative,
            quat_conjugate(self._rigid_body_rot_extend, w_last=True),
            w_last=True,
        )
        rotation_diff = quat_to_angle_axis(rot_diff)[0]
        error = rotation_diff**2
        return torch.exp(-error.mean(-1) / self.config.rewards.reward_tracking_sigma.motion_relative_body_orientation**2)

    def _reward_motion_global_body_angular_velocity_error_exp(self):
        error = torch.sum(self.dif_global_body_ang_vel**2, dim=-1)
        return torch.exp(
            -error.mean(-1) / self.config.rewards.reward_tracking_sigma.motion_global_body_angular_velocity**2
        )
    def _reward_undesired_contacts(self):
        if self.penalised_contact_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact_forces = self.simulator.contact_forces[:, self.penalised_contact_indices, :]
        is_contact = torch.norm(contact_forces, dim=-1) > self.config.rewards.threshold.undesired_contact_threshold
        return torch.sum(is_contact, dim=1)

    def _reward_limits_dof_pos(self):
        jpos_limits = self.simulator.dof_pos_limits
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        soft_factor = self.config.rewards.reward_limit.soft_dof_pos_limit
        lower_soft_limit = jpos_mean - 0.5 * jpos_range * soft_factor
        upper_soft_limit = jpos_mean + 0.5 * jpos_range * soft_factor

        jpos = self.simulator.dof_pos
        violation_min = (lower_soft_limit - jpos).clamp_min(0.0)
        violation_max = (jpos - upper_soft_limit).clamp_min(0.0)
        return (violation_min + violation_max).sum(1)


    def _reward_motion_global_body_linear_velocity_error_exp(self):
        error = torch.sum(self.dif_global_body_vel**2, dim=-1)
        return torch.exp(-error.mean(-1) / self.config.rewards.reward_tracking_sigma.motion_global_body_linear_velocity**2)

    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "maniskill":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
