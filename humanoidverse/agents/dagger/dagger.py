import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActorFixSigma
from humanoidverse.agents.modules.models import BaseModels
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseEnv
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

class DAgger(BaseAlgo):
    def __init__(self,
                 env: BaseEnv,
                 config,
                 log_dir=None,
                 device='cpu'):

        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _, _ = self.env.reset_all()

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
        self.learning_rate = self.config.learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss

    def setup(self):
        logger.info("Setting up Dagger")        
        
        actor_network_dict = dict(actor=self.config.network_dict['actor'])
        teacher_actor_network_dict = dict(actor=self.config.network_dict['teacher_actor'])
        teacher_actor_network_load_dict = dict(actor=self.config.network_load_dict['teacher_actor'])
        self.actor = PPOActorFixSigma(self.algo_obs_dim_dict, actor_network_dict, {}, self.num_act)
        self.gt_actor = PPOActorFixSigma(self.algo_obs_dim_dict, teacher_actor_network_dict, teacher_actor_network_load_dict, self.num_act)

        self.actor.to(self.device)
        self.gt_actor.to(self.device)
        # import ipdb; ipdb.set_trace()
        # print all keys in self.actor.parameters()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        logger.info(f"Setting up Storage")
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)

        self.storage.register_key('gt_actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('gt_actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('gt_action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('gt_action_sigma', shape=(self.num_act,), dtype=torch.float)
        
        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    
    def _eval_mode(self):
        self.actor.eval()

    def _train_mode(self):
        self.actor.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path)
            self.actor.load_state_dict(loaded_dict["actor_state_dict"])
            if self.load_optimizer:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.learning_rate = loaded_dict['optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Learning rate: {self.learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)
        
    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()

            # Jiawei: Need to return obs_dict to update the obs_dict for the next iteration
            # Otherwise, we will keep using the initial obs_dict for the whole training process
            obs_dict =self._rollout_step(obs_dict)

            mean_bc_loss = self._training_step()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': it,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'mean_bc_loss': mean_bc_loss,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': num_learning_iterations
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                actions = self.actor.act(obs_dict).detach()
                action_mean = self.actor.action_mean.detach()
                # action_sigma = self.actor.action_std.detach()
                # actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
                
                gt_actions = self.gt_actor.act(obs_dict).detach()
                gt_action_mean = self.gt_actor.action_mean.detach()
                # gt_action_sigma = self.gt_actor.action_std.detach()
                # gt_action_log_prob = self.gt_actor.get_actions_log_prob(gt_actions).detach().unsqueeze(1)
                
                assert len(actions.shape) == 2
                # assert len(actions_log_prob.shape) == 2
                assert len(action_mean.shape) == 2
                # assert len(action_sigma.shape) == 2

                policy_state_dict = dict(
                    actions=actions,
                    # actions_log_prob=actions_log_prob,
                    action_mean=action_mean,
                    # action_sigma=action_sigma,
                    gt_actions=gt_actions,
                    # gt_actions_log_prob=gt_action_log_prob,
                    gt_action_mean=gt_action_mean,
                    # gt_action_sigma=gt_action_sigma,
                )

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                    
                actions = policy_state_dict["actions"]
                obs_dict, rewards, dones, infos = self.env.step(actions)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                # rewards_stored = rewards.clone().unsqueeze(1)
                # if 'time_outs' in infos:
                #     rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                # assert len(rewards_stored.shape) == 2

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

            # returns, advantages = self._compute_returns(
            #     last_obs_dict=obs_dict,
            #     policy_state_dict=dict(values=self.storage.query_key('values'), 
            #     dones=self.storage.query_key('dones'), 
            #     rewards=self.storage.query_key('rewards'))
            # )
            # self.storage.batch_update_data('returns', returns)
            # self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)
 
    def _training_step(self):
        
        mean_bc_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_dict in generator:
            # Move everything to the device
            for obs_key in obs_dict.keys():
                obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

            # actions_batch = obs_dict['actions']
            # old_actions_log_prob_batch = obs_dict['actions_log_prob']
            # old_mu_batch = obs_dict['action_mean']
            # old_sigma_batch = obs_dict['action_sigma']
            
            gt_actions_batch = obs_dict['gt_actions']
            
            self.actor.act(obs_dict,)
            # actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            mu_batch = self.actor.action_mean
            # sigma_batch = self.actor.action_std
            # entropy_batch = self.actor.entropy
            
            bc_loss = torch.square(mu_batch - gt_actions_batch).mean()

           
            loss = self.config.bc_loss_coef * bc_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_bc_loss += bc_loss.item()
            # mean_value_loss += value_loss.item()
            # mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_bc_loss /= num_updates
        self.storage.clear()

        return mean_bc_loss

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))

        self.writer.add_scalar('Loss/learning_rate', self.learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/mean_bc_loss', log_dict['mean_bc_loss'], log_dict['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), log_dict['it'])
        self.writer.add_scalar('Perf/total_fps', fps, log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])

        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'BC loss:':>{pad}} {log_dict['mean_bc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state["actions"])
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            logger.info(f"{obs_key}, {sorted(self.env.config.obs.obs_dict[obs_key])}")
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    @property
    def inference_model(self):
        return self.actor

    def _get_inference_policy(self, device=None):
        self.actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference