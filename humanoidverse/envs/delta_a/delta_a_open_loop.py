import torch
import numpy as np
from pathlib import Path
import os
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking

class DeltaA_OpenLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        # import ipdb; ipdb.set_trace()
        super().__init__(config, device)
        self.delta_action_dof_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        self.detla_action_percentage_heatmaps = torch.zeros((self.simulator.num_envs, self.num_dofs)).to(device)
        self.delta_action_cnt = 1

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if hasattr(self.config, 'zero_delta_a'):
            if self.config.zero_delta_a:                
                actions *= 0.
                print("actions", actions)
                pass
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if self.config['add_extra_action']:
            motion_action = self.get_open_loop_action_at_current_timestep()
            # motion_action *= 0
            # print("motion_action", motion_action)
            # print("self.get_open_loop_action_at_current_timestep()", self.get_open_loop_action_at_current_timestep())
            motion_action *= self.config.robot.control.action_scale
        else:
            print("zeroing out motion_action") 
            motion_action = torch.zeros_like(actions_scaled)

        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR:
                actions_scaled[:, [i for i in range(actions_scaled.shape[1]) if i not in [4,5,10,11]]] *= 0
                
                # print("zeroing out non-anklePR actions")

        
        # add action_to_delta_a_heatmap
        
        self.delta_action_dof_heatmaps = self.delta_action_dof_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) * torch.abs(actions_scaled)
        self.detla_action_percentage_heatmaps = self.detla_action_percentage_heatmaps * self.delta_action_cnt / (1+self.delta_action_cnt) + 1/(1+self.delta_action_cnt) *  torch.abs(actions_scaled / motion_action)
        self.delta_action_cnt += 1


        print("self.delta_action_dof_heatmaps", self.delta_action_dof_heatmaps)
        print("self.detla_action_percentage_heatmaps", self.detla_action_percentage_heatmaps)
        # import ipdb; ipdb.set_trace()
        # print("motion_action", motion_action)
        # print("motion_action", motion_action)
        # print("motion_action", motion_action.shape)
        # print('actions', actions)
        # print('motion_action', motion_action)
        # import ipdb; ipdb.set_trace(

        # handcrafted perfect delta_a for 65Kp
        # perfect_delta_a_for_65Kp = (motion_action + self.default_dof_pos - self.simulator.dof_pos) * -0.35
        # actions_scaled = perfect_delta_a_for_65Kp

        if control_type=="P":
            # torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            torques = self._kp_scale * self.p_gains*(actions_scaled + motion_action + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques
        
    def _reward_minimal_action_norm(self):
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self.actions, dim=-1))


    def _reward_normalized_penalty_minimal_action_norm(self):
        # exp(-norm(actions))
        return torch.exp(-torch.norm(self.actions, dim=-1)) - 1

    def _reward_penalty_minimal_action_norm(self):
        # print("self.actions", self.actions)
        # print("torch.norm(self.actions, dim=-1)", torch.norm(self.actions, dim=-1))
        # # import ipdb; ipdb.set_trace()
        # # print max abs values of rewards
        # print("max abs values of rewards", torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)))
        # # throw out the reward if the action norm is too large, and tell me which action caused it
        # if torch.max(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1)) > 100000:
        #     print("action causing the reward to be thrown out", self.actions[torch.argmax(torch.abs(torch.exp(torch.norm(self.actions, dim=-1))-1))])
        #     import ipdb; ipdb.set_trace()
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        #     # return torch.tensor(0.0)
        # else:
        #     return torch.exp(torch.norm(self.actions, dim=-1))-1
        
        # clip the reward from -1000 to 1000
        return torch.exp(torch.norm(self.actions, dim=-1))-1
        # return torch.clip(torch.exp(torch.norm(self.actions, dim=-1))-1, -1000, 1000)
        
    def get_open_loop_action_at_current_timestep(self):
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_action = self._motion_lib.get_motion_actions(self.motion_ids, motion_times)

        return motion_action
    
    # NOTE: this is the perfect delta_a for 0.65Kp scenario
    def _get_perfect_delta_a(self):
        return (self.get_open_loop_action_at_current_timestep() * self.config.robot.control.action_scale + self.default_dof_pos - self.simulator.dof_pos) * -0.35

    def _get_obs_actions_open_loop(self):
        return self.get_open_loop_action_at_current_timestep()

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        # import ipdb; ipdb.set_trace()
        return self.simulator.dof_pos[:, [4,5,10,11]]
    
    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, [4,5,10,11]]

    def _get_obs_actions_ankle_pitch_roll(self):
        return self.actions[:, [4,5,10,11]]
    
    def _get_obs_actions_open_loop_ankle_pitch_roll(self):
        return self.get_open_loop_action_at_current_timestep()[:, [4,5,10,11]]
    

    

