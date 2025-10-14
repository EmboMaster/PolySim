import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

class PPODecoupled(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)

    def _init_config(self):
        super()._init_config()
        self.num_act = self.env.config.robot.lower_body_actions_dim

    def setup(self):
        logger.info("Setting up PPO_Decoupled")
        self._setup_models_and_optimizer()
        logger.info(f"Setting up Storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        self.actor = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)

        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                # actions = self.actor.act(obs_dict["actor_obs"]).detach()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                ## Get the lower body actions
                actions_lower_body = policy_state_dict["actions"]
                ## Get the upper body actions
                actions_upper_body = self.env.ref_upper_dof_pos
                ## Concatenate the lower and upper body actions
                actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), 
                dones=self.storage.query_key('dones'), 
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        # Update the upper body action scale
        if len(self.lenbuffer) == 0:
            mean_lenbuffer = 0
        else: mean_lenbuffer = statistics.mean(self.lenbuffer)
        if mean_lenbuffer > 0.9 * self.env.max_episode_length:
            self.env.action_scale_upper_body += 0.05
        else:
            self.env.action_scale_upper_body -= 0.01
        # Constraint the upper body action scale to (0, 1)
        self.env.action_scale_upper_body = max(0, min(self.env.action_scale_upper_body, 1))

        return obs_dict
    
    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        super()._logging_to_writer(log_dict, train_log_dict, env_log_dict)
        # Log the action scale for the upper body
        self.writer.add_scalar('Env/action_scale_upper_body', self.env.action_scale_upper_body, log_dict['it'])

    def env_step(self, actor_state):
        actions_lower_body = actor_state["actions"]
        actions_upper_body = self.env.ref_upper_dof_pos
        actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
        actor_state = {"actions": actions}
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state