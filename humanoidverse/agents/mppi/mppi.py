from typing import Dict, Any, Optional, Tuple
from multiprocessing import Process, Manager,shared_memory, Event, set_start_method, Queue
import struct
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import wandb

from humanoidverse.envs.base_task.base_task import BaseEnv
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.modules.world_models import BaseWorldModel
#from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
#from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.modules.world_models import SimWorldModel, MultiSimWorldModel

import time
import os
import statistics
from collections import deque
import copy
from tqdm import tqdm, trange
from hydra.utils import instantiate
from termcolor import colored



def softmax_update(weights, trajs, mean, std):
    # current in testing
    mean_new = torch.einsum("n,nij->ij", weights, trajs)
    std_new = std
    return mean_new, std_new

def cma_es_update(weights, trajs, mean, std):
    mean_new = (weights.view(-1, 1, 1) * trajs).sum(dim=0)
    error = trajs - mean
    std_new = torch.sqrt(torch.einsum("n,nij->ij", weights, error**2)).mean() * std
    std_new = torch.clamp(std_new, min=1e-3)
    return mean_new, std_new

def cem_update(weights, trajs, mean, std):
    idx = torch.argsort(weights)[-3:] 
    mean_new = torch.mean(trajs[idx], axis=0)
    std_new = std
    return mean_new, std_new


class CosAnnealScheduler:
    def __init__(self, x, T, config):
        self.x = x
        self.T = T
        self.min = config.min
        self.max = x
        self.t = 0

    def step(self, step=None):
        self.t += 1
        self.x = self.min + (self.max - self.min) * (1 + np.cos(self.t * np.pi / self.T)) / 2
        return self.x
    
    def reset(self):
        self.x = self.max
        self.t = 0
    

class LinearAnnealScheduler:
    def __init__(self, x, T, config):
        self.x = x
        self.T = T
        self.min = config.min
        self.max = x
        self.t = 0

    def step(self, step=None):
        self.t += 1
        self.x = self.min + (self.max - self.min) * (1 - self.t / self.T)
        return self.x
    
    def reset(self):
        self.x = self.max
        self.t = 0
    

class ExpAnnealScheduler:
    def __init__(self, x, T, config): #0.96 #1e-2
        self.x = x
        self.T = T
        self.t = 0
        self.alpha = config.alpha
        self.max = x
        self.min = config.min
        self.re_init = config.re_init # reinitialize std if reuse plan

    def step(self, step=None, std=None):
        """
        step: global step in simulation
        """
        self.t += 1
        if self.re_init > 0 and self.t==1:
            self.x = self.re_init
        self.x = max(self.min, self.x * self.alpha)
        return self.x
    
    def reset(self):
        self.x = self.max
        self.t = 0
    

class PathIntegral(BaseAlgo):
    dynamic_model: BaseWorldModel
    policy_prior: nn.Module = None

    def __init__(self, 
                 env_config,
                 config,
                 sim_config=None,
                 log_dir=None,
                 device='cpu'):
        """
        env_config: evaluation simulation environment configuration
        config: algorithm configuration
        sim_config: simulation environment configuration
        """
        super().__init__(env_config, config, device)
        self.eval_env_config = env_config
        if sim_config is not None:
            self.sim_config = sim_config
        self.log_dir = log_dir 
        self.device = device

        self.num_envs: int = self.env.config.num_envs
        self.act_dim = self.env.config.robot.actions_dim
        
        self.num_samples = self.config.num_samples
        self.temp = self.config.temp
        self.horizon = self.config.horizon 
        self.apply_plan_step = self.config.apply_plan_step
        self.wm_type = self.config.world_model.type
        self.reuse_plan = self.config.reuse_plan
        self.prev_plan = None
        if self.wm_type != "sim_parallel":
            self._get_dynamics()

        self.update_fn = {"mppi": softmax_update, 
                          "cma-es": cma_es_update, 
                          "cem": cem_update}[self.config.method]
        self.noise_scheduler = {"cos": CosAnnealScheduler,
                                "linear": LinearAnnealScheduler,
                                "exp": ExpAnnealScheduler}[self.config.noise_scheduler.type](x=self.config.init_std, T=self.config.num_iters, config=self.config.noise_scheduler)
        
    def setup(self):
        print("##### SETUP SAMPLE BASED MPC #####")

    def _get_dynamics(self): 
        wm_config = self.config.world_model
        if wm_config.type == "sim":
            # simulation model
            assert self.env.num_envs == self.num_samples + 1
            self.dynamic_model = SimWorldModel(wm_config, self.env, self.num_samples, self.device)
        elif wm_config.type == "sim_parallel":
            sim_config = self.sim_config
            #sim_config.headless = True # run in headless mode on rollout
            print(f"ready for creating {sim_config.config.num_envs} sim...")#
            self.dynamic_model = MultiSimWorldModel(wm_config, sim_config, self.num_samples, self.config.command, self.device)
        else:
            raise NotImplementedError

    def _get_rollouts(self, init_state, init_buf=None, rollout_acts=None, rollout_policy=None):
        self.dynamic_model.reset(init_state, init_buf)
        use_policy = rollout_policy is not None
        if rollout_acts is None:
            assert rollout_policy is not None
            rollout_acts = torch.zeros((self.num_samples, self.horizon, self.act_dim), dtype=torch.float32, device=self.device)
 
        rews = torch.zeros((self.num_samples, self.horizon), dtype=rollout_acts.dtype, device=rollout_acts.device)
        last_obs = init_state["obs"]
        states = torch.zeros((self.num_samples, self.horizon + 1, last_obs.shape[-1]), dtype=rollout_acts.dtype, device=rollout_acts.device)
        states[:, 0] = last_obs # get initial observation
        n_done_mask = torch.ones(self.num_samples, device=self.device).bool()
        valid_len = torch.zeros(self.num_samples, device=self.device)
        for t in range(self.horizon):
            valid_len += n_done_mask.float()
            if use_policy:
                rollout_acts[:, t] = rollout_policy(last_obs)
            next_state, rew, not_done = self.dynamic_model.step(rollout_acts[:, t])
            states[:, t+1][n_done_mask] = next_state[n_done_mask]
            rews[:, t][n_done_mask] = rew[n_done_mask]
            last_obs = next_state
            n_done_mask = n_done_mask * not_done
            if n_done_mask.sum() == 0: # all done
                break

        rets = rews.sum(dim=-1)
        return states, rollout_acts, rets

    def _update_once(self, init_state, init_buf, mean, std, step=None):
        epsilon = torch.randn((self.num_samples, self.horizon, self.act_dim), device=self.device) * std
        trajs = torch.clamp(epsilon + mean, - np.pi, np.pi)
        _, _, rets = self._get_rollouts(init_state, init_buf, trajs)
        logp0 = (rets - rets.mean()) / (rets.std() + 1e-9) / self.temp
        weights = torch.softmax(logp0, dim=0)
        mean_new, std_new = self.update_fn(weights, trajs, mean, std)
        if self.config.method != "cma-es":
            std_new = self.noise_scheduler.step(step)
        return rets, mean_new, std_new
    
    def _init_trajectory(self, init_state):
        # initialize mean and std of action trajectory
        if self.policy_prior is not None:
            _, init_mean, _ = self._get_rollouts(init_state=init_state, 
                                           rollout_acts=None, 
                                           rollout_policy=self.policy_prior)
        elif self.prev_plan is not None and self.reuse_plan:
            init_mean = torch.zeros((self.num_samples, self.horizon, self.act_dim), device=self.device)
            init_mean[:, :-self.apply_plan_step] = self.prev_plan
            init_mean[:, -self.apply_plan_step:] = self.prev_plan[-1].unsqueeze(0).unsqueeze(0).repeat(self.num_samples, self.apply_plan_step, 1) # keep last action
            init_std = self.config.init_std * torch.clamp(torch.arange(1, self.horizon+1, 1, device=self.device).unsqueeze(0).unsqueeze(-1) / self.horizon, 0.1, 1.0)
        else:
            init_mean = torch.zeros((self.num_samples, self.horizon, self.act_dim), device=self.device)
            init_std = self.config.init_std * torch.ones((self.act_dim,), device=self.device)
        return init_mean, init_std

    def planning(self, init_state, init_buf=None, step=None):
        mean, std = self._init_trajectory(init_state)
        tbar = tqdm(range(self.config.num_iters), desc="Processing", leave=True)
        anneal_steps = 0 
        for _ in tbar:
            rets, mean, std = self._update_once(init_state, init_buf, mean, std, step)
            anneal_steps += 1
            tbar.set_postfix({"max_return": rets.max().item(),
                                "mean_return": rets.mean().item(),
                                "noise std": std})
            if rets.mean() >  self.config.plan_end_ret_mean* self.horizon or rets.max() > self.config.plan_end_ret_max * self.horizon:
                break # early stop if reach the target return    

        self.noise_scheduler.reset()
        planning_info = {"anneal_steps": anneal_steps,
                         "final_max_return": rets.max().item(),
                         "final_mean_return": rets.mean().item(), 
                         "final_noise_std": std}
        return mean, planning_info
    
    def act_process(self, manager_dict, log_queue):
        self._get_dynamics() # get dynamics model
        init_buf = self.dynamic_model.env.get_mppi_buffers([0]) # get initial buffer for sim envs

        dim_info = self.dynamic_model.get_env_dim()
        env_num_dof = dim_info['dof_shape'][1]
        env_root_state_dim = dim_info['root_states_shape'][-1]
        obs_dim = dim_info['obs_dim']
        
        plan_ready = manager_dict['plan_ready']
        sim_ready = manager_dict['sim_ready']
        mem_ready = manager_dict['mem_ready']

        mem_ready.wait()  # Wait for memory to be ready
        try:
            state_shm = shared_memory.SharedMemory(name="state_shm")
            state_buffer = state_shm.buf
        except FileNotFoundError:
            print("State shared memory 'state_shm' not found.")
            exit()
        try:
            ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
            ctrl_buffer = ctrl_shm.buf
        except FileNotFoundError:
            print("Could not create control shared memory 'ctrl_shm'.")
            exit()
        try:
            buf_shim = shared_memory.SharedMemory(name="buf_shm")
            buf_buffer = buf_shim.buf
        except FileNotFoundError:
            print("Could not create buffer shared memory 'buf_shm'.")
            exit()

        try:
            step=0
            while True:
                sim_ready.wait()
                sim_ready.clear()
                # get reset states and buffer from shared memory
                buffered_state = state_buffer[:]
                t_real = struct.unpack("d", buffered_state[0:8])[0]
                state = {"dof_states": torch.frombuffer(buffered_state[8 : 8 + env_num_dof * 2 * 8], dtype=torch.float64).to(torch.float32).view(1, env_num_dof, 2).to(self.device),
                        "root_states": torch.frombuffer(buffered_state[8 + env_num_dof * 2 * 8 : 8 + env_num_dof * 2 * 8 + env_root_state_dim * 8], dtype=torch.float64).to(torch.float32).view(1, -1).to(self.device), 
                        "obs": torch.frombuffer(buffered_state[8 + env_num_dof * 2 * 8 + env_root_state_dim * 8 :], dtype=torch.float64).to(torch.float32).view(1, obs_dim).to(self.device)} # use real states for simulation world model
                buffered_buf = buf_buffer[:]
                tgt_buf = {}
                buf_size = 0
                for k, v in init_buf.items():
                    tgt_buf[k] = torch.frombuffer(buffered_buf[buf_size : buf_size+v.numel() * 8], dtype=torch.float64).view(v.shape).to(self.device)
                    buf_size += v.numel() * 8

                # plannning
                ctrl_time = t_real # TODO: get time from state
                curr_plan, curr_info = self.planning(state, tgt_buf, step)
                ctrl_act = curr_plan[:self.apply_plan_step].to(torch.float64).cpu().numpy()
                if self.reuse_plan:
                    self.prev_plan = curr_plan[self.apply_plan_step:] # store previous actions for next iteration

                ctrl_shm_size = 8 + self.act_dim * self.apply_plan_step * 8 # time + action
                ctrl_data = bytearray(ctrl_shm_size)
                struct.pack_into("d", ctrl_data, 0, ctrl_time)
                struct.pack_into("d" * self.act_dim * self.apply_plan_step, ctrl_data, 8, *ctrl_act.flatten())
                ctrl_buffer[:] = ctrl_data

                for k, v in curr_info.items():
                    log_queue.put({"tag": f"Planning/{k}", "value": v, "step": step})
                
                plan_ready.set()
                step += 1
        
        except KeyboardInterrupt:
            pass
        finally:
            state_shm.close()
            ctrl_shm.close()
            buf_shim.close()

    @torch.no_grad()
    def eval_process(self, manager_dict, log_queue):

        plan_ready = manager_dict['plan_ready']
        sim_ready = manager_dict['sim_ready']
        mem_ready = manager_dict['mem_ready']   

        self.env = instantiate(self.eval_env_config, device=self.device) # create evaluation environment
        self.env.set_is_evaluating(self.config.command)
        obs = self.env.reset_all()["actor_obs"]
        init_buf = self.env.get_mppi_buffers([0]) # get buffer for the environment
        # cleaning previous shared memory
        try:
            state_shm = shared_memory.SharedMemory(name="state_shm")
            state_shm.close()
            state_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
            ctrl_shm.close()
            ctrl_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            buf_shm = shared_memory.SharedMemory(name="buf_shm")
            buf_shm.close()
            buf_shm.unlink()
        except FileNotFoundError:
            pass

        # Shared Memory for control inputs
        ctrl_shm_size = 8 + self.act_dim * self.apply_plan_step * 8 # time + action
        ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm", create=True, size=ctrl_shm_size)
        ctrl_buffer = ctrl_shm.buf
        # Shared Memory for state variables
        env_num_dof = self.env.num_dofs
        env_root_state_dim = self.env.robot_root_states.shape[-1]  
        obs_dim = obs.shape[-1]
        state_shm_size = 8 + env_num_dof * 2 * 8 + env_root_state_dim * 8 + obs_dim * 8  # time + dof_state + root_state + obs
        state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=state_shm_size
        )
        state_buffer = state_shm.buf
        # Shared Memory for buffer
        buf_shm_size = 0
        for k, v in init_buf.items():
            buf_shm_size += v.numel() * 8 
        buf_shm = shared_memory.SharedMemory(
            name="buf_shm", create=True, size=buf_shm_size
        )
        buf_buffer = buf_shm.buf
        # Signal that memory is ready
        mem_ready.set()  

        t0 = time.time()
        step = 0
        stored_actions = []
        stored_states = []  
        start_reset_state = None
        start_reset_buf = None
        try:
            try:
                cum_rew = 0
                done = False
                while not done:
                    t_real = time.time() - t0 
                    curr_state = {"dof_states": copy.deepcopy(self.env.dof_state.view(1, self.env.num_dof, 2)),
                                "root_states": copy.deepcopy(self.env.robot_root_states),
                                "obs": copy.deepcopy(obs)} # use real states for simulation world model
                    
                    # packing state data into shared memory
                    state_data = bytearray(state_shm_size)
                    struct.pack_into("d", state_data, 0, t_real)
                    struct.pack_into("d" * (env_num_dof * 2), state_data, 8, *curr_state["dof_states"].to(torch.float64).cpu().numpy().flatten())
                    struct.pack_into("d" * env_root_state_dim, state_data, 8 + env_num_dof * 2 * 8, *curr_state["root_states"].to(torch.float64).cpu().numpy().flatten())
                    struct.pack_into("d" * obs_dim, state_data, 8 + env_num_dof * 2 * 8 + env_root_state_dim * 8, *curr_state["obs"].to(torch.float64).cpu().numpy().flatten())
                    state_buffer[:] = state_data

                    curr_buf = self.env.get_mppi_buffers([0]) # get buffer for the environment
                    buf_data = bytearray(buf_shm_size)
                    buf_size = 0
                    for k, v in curr_buf.items():
                        struct.pack_into("d" * v.numel(), buf_data, buf_size, *v.to(torch.float64).cpu().numpy().flatten()) # pack buffer data
                        buf_size += v.numel() * 8
                    buf_buffer[:] = buf_data


                    if step == 0:
                        start_reset_state = curr_state
                        start_reset_buf = curr_buf

                    sim_ready.set()  # Signal that simulation is ready for the next plan
                    plan_ready.wait()  # Wait for the planning to be ready
                    plan_ready.clear()

                    ctrl_data = ctrl_buffer[:]
                    ctrl_time = struct.unpack("d", ctrl_data[0:8])[0]
                    actions = torch.frombuffer(ctrl_data[8:], dtype=torch.float64).to(torch.float32).view(self.apply_plan_step, self.act_dim).to(self.device)

                    for i in range(self.apply_plan_step):
                        stored_actions.append(actions[[i]].cpu().numpy())
                        stored_states.append(curr_state)
                        nxt_full_ob, reward, dones, info = self.env.step(actions[[i]])
                        done = dones[0]
                        cum_rew += reward[0]
                        t_curr = time.time() - t0
                        step += 1
                        print(f"at step {step} cumulated reward {cum_rew}; planning time {t_curr:.2f} seconds")
                        log_queue.put({"tag": "Eval/reward_per_step", "value": reward[0], "step": step})
                        log_queue.put({"tag": "Eval/cumulative_reward", "value": cum_rew, "step": step})
                        log_queue.put({"tag": "Eval/eval_time", "value": t_curr, "step": step})
                        if done:
                            break

                    if step >= self.config.eval_steps:
                        break
                        
                if self.config.log_actions:
                    pkl.dump(stored_actions, open(f"{self.log_dir}/data/stored_actions_{self.config.eval_steps}step_{self.config.command}.pkl", "wb"))  
                if self.config.log_states:
                    pkl.dump(stored_states, open(f"{self.log_dir}/data/stored_states_{self.config.eval_steps}step_{self.config.command}.pkl", "wb"))
                print(f"Cumulative reward after {step}steps rollout:", cum_rew)

                # ## Vis loop
                # time.sleep(1)
                # print("\nstaring visualization loop")
                # self.env.set_is_evaluating(self.config.command)
                # self.env.reset_all()
                # self.env.reset_envs_idx(torch.tensor([0]).to(self.device), targt_states=start_reset_state, targt_buf=start_reset_buf)
                # cum_rew = 0
                # done = False
                # for i in range(len(stored_actions)):
                #     nxt_full_ob, reward, dones, info = self.env.step(torch.from_numpy(stored_actions[i]).to(self.device))
                #     done = dones[0]
                #     cum_rew += reward[0]
                #     print(f"step{i+1} cumulative reward {cum_rew}")
                #     if done:
                #         print("early stop")
                #         break
                # print("cumulative reward:", cum_rew)

            except KeyboardInterrupt:
                pass

        finally:
            # Clean up shared memory
            state_shm.close()
            state_shm.unlink()
            ctrl_shm.close()
            ctrl_shm.unlink()
            buf_shm.close()
            buf_shm.unlink()

    def log_process(self, wandb_config, manager_dict, log_queue):
        if wandb_config is not None:
            wandb.init(project=wandb_config["project"], 
                    entity=wandb_config["entity"],
                    name=wandb_config["name"],
                    sync_tensorboard=wandb_config["sync_tensorboard"],
                    config=wandb_config["config"],
                    dir=wandb_config["dir"])
        
        writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        while True:
            log_data = log_queue.get()
            if log_data is None:  # End logging when receiving None
                break
            writer.add_scalar(log_data["tag"], log_data["value"], log_data["step"])
        writer.close()
    
    def evaluate_policy(self, wandb_config=None):
        with Manager() as manager:
            manager_dict = {
                'plan_ready': manager.Event(),
                'sim_ready': manager.Event(),
                'mem_ready': manager.Event(),
            }
            for k, v in manager_dict.items():
                v.clear()

            log_queue = Queue() # Queue for logging
            # Start the evaluation loop in a separate process
            eval_proc = Process(target=self.eval_process, args=(manager_dict, log_queue))
            act_proc = Process(target=self.act_process, args=(manager_dict, log_queue))
            log_proc = Process(target=self.log_process, args=(wandb_config, manager_dict, log_queue))

            eval_proc.start()
            act_proc.start()
            log_proc.start()

            eval_proc.join()
            act_proc.join()
            log_queue.put(None)
            log_proc.join()