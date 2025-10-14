import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
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


from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from pathlib import Path
from omegaconf import OmegaConf
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.base_algo.base_algo import BaseAlgo 
from hydra.utils import instantiate


class PPODeltaA(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)

        if config.policy_checkpoint is not None:
            has_config = True
            checkpoint = Path(config.policy_checkpoint)
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    has_config = False
                    logger.error(f"Could not find config path: {config_path}")

            if has_config:
                logger.info(f"Loading training config file from {config_path}")
                with open(config_path) as file:
                    policy_config = OmegaConf.load(file)

                if policy_config.eval_overrides is not None:
                    policy_config = OmegaConf.merge(
                        policy_config, policy_config.eval_overrides
                    )
            
        pre_process_config(policy_config)
        
        self.loaded_policy: BaseAlgo = instantiate(policy_config.algo, env=env, device=device, log_dir=None)
        self.loaded_policy.algo_obs_dim_dict = policy_config.env.config.robot.algo_obs_dim_dict
        self.loaded_policy.setup()

        self.loaded_policy.load(config.policy_checkpoint)

        
        # import ipdb; ipdb.set_trace()
        self.loaded_policy._eval_mode()

        for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
            param.requires_grad = False
            # print(f"Parameter name: {name}, Parameter: {param}, Requires Grad: {param.requires_grad}")
        
        # import ipdb; ipdb.set_trace()   
        self.loaded_policy.eval_policy = self.loaded_policy._get_inference_policy()
        



    def _rollout_step(self, obs_dict):
        # import ipdb; ipdb.set_trace()
        # self._eval_mode()
        # self.eval_policy = self._get_inference_policy()
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
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                
                ################ inference the policy ################
                policy_output = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()
                actor_state["actions_closed_loop"] = policy_output

                ######################################################

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

        return obs_dict
    

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actions_closed_loop = self.loaded_policy.eval_policy(actor_state['obs']['closed_loop_actor_obs']).detach()

        actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
