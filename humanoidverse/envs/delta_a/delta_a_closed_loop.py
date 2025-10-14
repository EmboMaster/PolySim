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
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float

class DeltaA_ClosedLoop(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        # import ipdb; ipdb.set_trace()
        super().__init__(config, device)

        
        
        self.closed_loop_actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)

    def _init_buffers(self):
        super()._init_buffers()
        if self.config.domain_rand.cotrain_with_without_delta_a:
            without_delta_a_ratio = self.config.domain_rand.without_delta_a_ratio
            self.with_delta_a_or_not = torch.rand(self.num_envs, device=self.device) > without_delta_a_ratio
            self.with_delta_a_or_not = self.with_delta_a_or_not.unsqueeze(1)
        
        self.delta_a_scale = torch.ones(self.num_envs, device=self.device, requires_grad=False)
        if self.config.domain_rand.rescale_delta_a:
            self.delta_a_scale = torch_rand_float(self.config.domain_rand.delta_a_scale_range[0], self.config.domain_rand.delta_a_scale_range[1], (self.num_envs, self.num_dofs), device=self.device)

    def _episodic_domain_randomization(self, env_ids):
        super()._episodic_domain_randomization(env_ids)
        
        if self.config.domain_rand.rescale_delta_a:
            self.delta_a_scale[env_ids] = torch_rand_float(self.config.domain_rand.delta_a_scale_range[0], self.config.domain_rand.delta_a_scale_range[1], (len(env_ids), self.num_dofs), device=self.device)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if self.config['add_extra_action']:
            motion_action = self.get_closed_loop_action_at_current_timestep()
        else:
            motion_action = torch.zeros_like(actions_scaled)
        

        if self.config.domain_rand.cotrain_with_without_delta_a:

            motion_action = motion_action * self.with_delta_a_or_not
        if self.config.domain_rand.rescale_delta_a:
            motion_action = motion_action * self.delta_a_scale

        # import ipdb; ipdb.set_trace()
        if hasattr(self.config.domain_rand, 'action_noise'): # domain_rand.action_noise_percentage=0.05
            if self.config.domain_rand.action_noise:
                # import ipdb; ipdb.set_trace()
                print("adding action noise")
                print("noise percentage", self.config.domain_rand.action_noise_percentage)
                action_noise = (torch.rand_like(actions_scaled)*2.-1.) * 1.0 * self.config.domain_rand.action_noise_percentage
                actions_scaled += action_noise
        
        
        if hasattr(self.config, 'delta_a_gradient_search'):
            if self.config.delta_a_gradient_search:
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 2000
                    torch.set_grad_enabled(True)

                    # import ipdb; ipdb.set_trace()

                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs

                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()

                    norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                    new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent):
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        loss_fn = new_best_action + self.loaded_extra_policy.eval_policy(new_input_for_deltaA) - norminal_policy_action

                        loss_fn = loss_fn.norm(dim=1)
                        loss_fn.backward()
                        with torch.no_grad():
                            new_best_action -= 0.0002 * new_best_action.grad
                        new_best_action.grad.zero_()
                        print("iter", i)
                        print("loss_fn", loss_fn)
                        print("new_best_action", new_best_action)
                        print("-----------------------------------------")
                        

                    torch.set_grad_enabled(False)
                
                    # replace action_scaled with new_best_action *= self.config.robot.control.action_scale
                    actions_scaled = new_best_action * self.config.robot.control.action_scale

        
        if hasattr(self.config, 'delta_a_fixed_point_iteration'):
            if self.config.delta_a_fixed_point_iteration:
                if hasattr(self, 'loaded_extra_policy') and self.current_closed_loop_actor_obs is not None:
                    iter_for_gradient_descent = 10



                    current_closed_loop_actor_obs = self.current_closed_loop_actor_obs

                    fixed_part_obs_for_deltaA = current_closed_loop_actor_obs[:, :-23].detach().clone()

                    norminal_policy_action = current_closed_loop_actor_obs[:, -23:].detach().clone()
                    new_best_action = current_closed_loop_actor_obs[:, -23:].detach().clone().requires_grad_(True)

                    for i in range(iter_for_gradient_descent):
                        new_input_for_deltaA = torch.cat([fixed_part_obs_for_deltaA, new_best_action], dim=1)
                        new_best_action =  norminal_policy_action - self.loaded_extra_policy.eval_policy(new_input_for_deltaA)

                        print("iter", i)
                        print("new_best_action", new_best_action)
                        print("-----------------------------------------")
                        


                
                    # replace action_scaled with new_best_action *= self.config.robot.control.action_scale
                    actions_scaled = new_best_action * self.config.robot.control.action_scale



        if hasattr(self.config, 'anklePR'):
            if self.config.anklePR:
                motion_action[:, [i for i in range(actions_scaled.shape[1]) if i not in [4,5,10,11]]] *= 0
                print("zeroing out non-anklePR actions")



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
        
    def get_closed_loop_action_at_current_timestep(self):
        return self.actions_closed_loop.clone()

    def _get_obs_actions_closed_loop(self):
        return self.actions_closed_loop.clone()
    
    def _get_obs_actions_sim2real_policy(self):
        return self.actions

    def _get_obs_dof_pos_ankle_pitch_roll(self):
        # import ipdb; ipdb.set_trace()
        return self.simulator.dof_pos[:, [4,5,10,11]]
    
    def _get_obs_dof_vel_ankle_pitch_roll(self):
        return self.simulator.dof_vel[:, [4,5,10,11]]

    def _get_obs_actions_closed_loop_ankle_pitch_roll(self):
        return self.actions_closed_loop.clone()[:, [4,5,10,11]]
    
    def _get_obs_actions_sim2real_policy_ankle_pitch_roll(self):
        return self.actions[:, [4,5,10,11]]
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]
        self.actions_closed_loop = actor_state["actions_closed_loop"]
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions_closed_loop = torch.clip(self.actions_closed_loop, -clip_action_limit, clip_action_limit).to(self.device)


        # --------------------------- UNCOMMENT THIS FOR CURRENT_CLOSED_LOOP_ACTOR_OBS ---------------------------
        # if hasattr(self, 'current_closed_loop_actor_obs') and 'current_closed_loop_actor_obs' in actor_state.keys():
        #     self.current_closed_loop_actor_obs = actor_state['current_closed_loop_actor_obs']
        #     print("current_closed_loop_actor_obs found in self, set it to actor_state['current_closed_loop_actor_obs']")
        # else:
        #     if 'current_closed_loop_actor_obs' in actor_state.keys():
        #         setattr(self, 'current_closed_loop_actor_obs', actor_state['current_closed_loop_actor_obs'])
        #         print("current_closed_loop_actor_obs not found in self, set it to actor_state['current_closed_loop_actor_obs']")
        #     else:
        #         setattr(self, 'current_closed_loop_actor_obs', None)
        #         print("current_closed_loop_actor_obs not found in actor_state")
        # --------------------------------------------------------------------------------------------------------


        
        # import ipdb; ipdb.set_trace()
        # print('actions in env.step: ', actions) 
        # print('closed_loop_actions in env.step: ', actor_state['actions_closed_loop'])
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()


        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def reset_all(self):
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actions_closed_loop = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False) 
        actor_state = {}
        actor_state["actions"] = actions
        actor_state["actions_closed_loop"] = actions_closed_loop
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict
    
    