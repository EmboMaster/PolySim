import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
from humanoidverse.utils.helpers import pre_process_config

class DeltaDynamics_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeltaDynamics_NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class DeltaDynamicsModel(BaseAlgo):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        
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
            # import ipdb; ipdb.set_trace()
            # for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
            #     if name == '6.bias':
            #         print(f"Parameter name: {name}, Parameter:  {param}")
            self.loaded_policy.load(config.policy_checkpoint)

            
            # import ipdb; ipdb.set_trace()
            self.loaded_policy._eval_mode()

            for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
                param.requires_grad = False
                # print(f"Parameter name: {name}, Parameter: {param}, Requires Grad: {param.requires_grad}")
            
            # import ipdb; ipdb.set_trace()   
            self.loaded_policy.eval_policy = self.loaded_policy._get_inference_policy()
        
        
        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        self._init_config()
        
        self.delta_dynamics_loss = nn.MSELoss()
        
        self.delta_dynamics_path = config.delta_dynamics_path
        # if self.delta_dynamics_path is not None:
        #     self.load(self.delta_dynamics_path)

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # Logging related Config

        self.save_interval = self.config.save_interval
        # Training related Config
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.num_learning_iterations = self.config.num_learning_iterations
        self.init_at_random_ep_len = self.config.init_at_random_ep_len

        # Algorithm related Config

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss

    
    def load(self, path):
        checkpoint = torch.load(path)
        self.delta_dynamics.load_state_dict(checkpoint['delta_dynamics'])
        self.delta_dynamics_optimizer.load_state_dict(checkpoint['delta_dynamics_optimizer'])
        self.delta_dynamics.to(self.device)
        logger.info(f"Loaded delta dynamics model from {path}")
        
        # Load pretrained policy
        ...
        
    def save(self, path):
        torch.save({
            'delta_dynamics': self.delta_dynamics.state_dict(),
            'delta_dynamics_optimizer': self.delta_dynamics_optimizer.state_dict()
        }, path)
        logger.info(f"Saved delta dynamics model to {path}")
        
    def setup(self):
        # import ipdb; ipdb.set_trace()
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        
    def _setup_models_and_optimizer(self):
        self.input_dim = self.env.get_input_dim()
        self.output_dim = self.env.get_output_dim()  

        self.delta_dynamics = DeltaDynamics_NN(self.input_dim, self.output_dim)
        self.delta_dynamics.to(self.device)
        self.delta_dynamics_optimizer = torch.optim.Adam(self.delta_dynamics.parameters(), lr=1e-3)
        
    def _eval_mode(self):
        self.delta_dynamics.eval()
        
    def _train_mode(self):
        self.delta_dynamics.train()
        
    def learn(self):
        self._train_mode()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        for it in range(self.num_learning_iterations):
            print('Iteration: ', it)
            if it % 1000 == 0: 
                # TODO: Change hardcoded number
                self.env.resample_motion()
                
            
            obs_dict = self.env.reset_all()

            self.delta_dynamics_optimizer.zero_grad()
            # obs_dict = self._rollout_step(obs_dict)

            # collect gradients
            loss = 0
            for i in range(self.num_steps_per_env):
                obs = obs_dict['delta_dynamics_input_obs']
                pred_delta = self.delta_dynamics(obs)
                # print('True')
                # parse tensor to dict
                delta_state_items = self.env.parse_delta(pred_delta.clone(), 'pred')
                
                obs_dict, rew_buf, reset_buf, extras = self.env.step(dict(on_policy=False))
                
                pred_state = self.env.update_delta(delta_state_items)
                
                # assemble dict to tensor
                pred = self.env.assemble_delta(pred_state)
                target = obs_dict['delta_dynamics_motion_state_obs']
                # loss += self.delta_dynamics_loss(pred, target)
                # print('Loss: ', loss)
                
                target_state_items = self.env.parse_delta(target.clone(), 'target')
                # calculate losses for different components
                loss_dof_pos = self.delta_dynamics_loss(pred_state['dof_pos'], target_state_items['motion_dof_pos']) 
                loss_dof_vel = self.delta_dynamics_loss(pred_state['dof_vel'], target_state_items['motion_dof_vel']) 
                loss_base_pos_xyz = self.delta_dynamics_loss(pred_state['base_pos_xyz'], target_state_items['motion_base_pos_xyz'])  
                loss_base_lin_vel = self.delta_dynamics_loss(pred_state['base_lin_vel'], target_state_items['motion_base_lin_vel'])  
                loss_base_ang_vel = self.delta_dynamics_loss(pred_state['base_ang_vel'], target_state_items['motion_base_ang_vel'])
                loss_base_quat = self.delta_dynamics_loss(pred_state['base_quat'], target_state_items['motion_base_quat'])   
                
                loss = loss_dof_pos + loss_dof_vel + loss_base_pos_xyz + loss_base_lin_vel + loss_base_ang_vel + loss_base_quat
                
            # update model
            self.writer.add_scalar('Loss', loss.item(), it)
            self.writer.add_scalar('Loss_dof_pos', loss_dof_pos.item(), it)
            self.writer.add_scalar('Loss_dof_vel', loss_dof_vel.item(), it)
            self.writer.add_scalar('Loss_base_pos_xyz', loss_base_pos_xyz.item(), it)
            self.writer.add_scalar('Loss_base_lin_vel', loss_base_lin_vel.item(), it)
            self.writer.add_scalar('Loss_base_ang_vel', loss_base_ang_vel.item(), it)
            self.writer.add_scalar('Loss_base_quat', loss_base_quat.item(), it)
            
            loss.backward()
            self.delta_dynamics_optimizer.step()
            if it % self.save_interval == 0:
                logger.info(f"Iteration: {it}, Loss: {loss.item()}")
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
    
    def _pre_eval_env_step(self, actor_state: dict):
        # import ipdb; ipdb.set_trace()
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actor_state.update({"actions": actions})
        pred_delta = self.delta_dynamics(actor_state["obs"]['delta_dynamics_input_obs'])
        delta_state_items = self.env.parse_delta(pred_delta.clone(), 'pred')
        actor_state['delta_state_items'] = delta_state_items
        return actor_state
    
    def eval_env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state
    
    def _post_eval_env_step(self, actor_state):
        
        # overwrite state with delta state
        delta_state_items = actor_state['delta_state_items']
        pred_state = self.env.update_delta(delta_state_items)
        
        # re-compute observations
        self.env._pre_compute_observations_callback()
        self.env._compute_observations()
        self.env._post_compute_observations_callback()
        # actor_state.update({"obs": self.obs_buf_dict})
        return actor_state
    
    
    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    
    @torch.no_grad()
    def evaluate_policy(self):
        # TODO: turn off gradient
        step = 0
        self.eval_policy = self.loaded_policy.eval_policy
        obs_dict = self.env.reset_all()
        actor_state = self._create_actor_state()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        init_delta_dynamics_input_obs = torch.zeros(self.env.num_envs, self.input_dim, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions, "on_policy": True, "delta_dynamics_input_obs": init_delta_dynamics_input_obs})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.eval_env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
            
    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict