import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.modules.encoder_modules import Estimator

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

class PPOEstimator(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()

        self.estimator = Estimator(self.algo_obs_dim_dict,
                                    self.config.module_dict.estimator).to(self.device)
        self.optimizer_estimator = torch.optim.Adam(self.estimator.parameters(),
                                                    lr=self.config.estimator_learning_rate)
        
    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            self.estimator.load_state_dict(loaded_dict["estimator_model_state_dict"])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.optimizer_estimator.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.estimator_learning_rate = loaded_dict['estimator_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate, self.estimator_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
                logger.info(f"Estimator Learning rate: {self.estimator_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]
        
    def set_learning_rate(self, actor_learning_rate, critic_learning_rate, estimator_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.estimator_learning_rate = estimator_learning_rate
    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'estimator_model_state_dict': self.estimator.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.optimizer_estimator.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def _eval_mode(self):
        super()._eval_mode()
        self.estimator.eval()

    def _train_mode(self):
        super()._train_mode()
        self.estimator.train()

    def _setup_storage(self):
        super()._setup_storage()
        self.storage.register_key('estimator_output', shape=(self.algo_obs_dim_dict["estimator_target"],), dtype=torch.float)
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['Estimator_Loss'] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = super()._update_algo_step(policy_state_dict, loss_dict)
        loss_dict = self._update_estimator(policy_state_dict, loss_dict)
        return loss_dict
    
    def _actor_act_step(self, obs_dict):
        long_history = obs_dict["long_history_for_estimator"]
        estimator_output = self.estimator(long_history).detach()
        actor_obs = obs_dict["actor_obs"]
        actor_input = torch.cat([actor_obs, estimator_output], dim=-1)
        return self.actor.act(actor_input)

    def _update_estimator(self, policy_state_dict, loss_dict):
        estimator_target = policy_state_dict["estimator_target"]
        estimator_output = self.estimator(policy_state_dict["long_history_for_estimator"])
        estimator_loss = F.mse_loss(estimator_output, estimator_target)

        self.optimizer_estimator.zero_grad()
        estimator_loss.backward()
        nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
        self.optimizer_estimator.step()

        loss_dict['Estimator_Loss'] = estimator_loss
        return loss_dict

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################
    def _pre_eval_env_step(self, actor_state: dict):
        estimator_output = self.estimator(actor_state["obs"]['long_history_for_estimator'])
        actor_state.update({"estimator_output": estimator_output})
        input_for_actor = torch.cat([actor_state["obs"]['actor_obs'], estimator_output], dim=-1)
        actions = self.eval_policy(input_for_actor)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
    
    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critic": self.critic,
            "estimator": self.estimator
        }