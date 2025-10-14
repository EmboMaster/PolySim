import os
import sys
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from typing import Dict, Any, List
import numpy as np
from loguru import logger
from humanoidverse.envs.base_task.base_task import BaseTask
import rpc_api
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class EnvClient:
    """
    Optimized distributed environment client communicating with multiple servers via RPC.
    Aggregates results and provides a unified interface compatible with BaseTask for PPO training.
    """
    
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device
        
        # RPC configuration
        self.rpc_config = config.rpc
        self.server_names = getattr(config.rpc, 'server_names', ["isaacsim", "isaacgym"]) 
        self.client_name = getattr(config.rpc, 'client_name', "ppo_trainer")    
        
        # Environment configuration
        self.num_envs = config.num_envs
        self.num_envs_list = config.num_envs_list
        self.dim_obs = config.robot.policy_obs_dim
        self.dim_critic_obs = config.robot.critic_obs_dim
        self.dim_actions = config.robot.actions_dim
        self.max_episode_length_s = config.max_episode_length_s
        self.max_episode_length = config.max_episode_length_s
        self.episode_length_buf = {}
        
        # Calculate environment distribution across servers
        self.envs_per_server = self._calculate_envs_distribution()
        
        # Preallocate tensors and initialize caches
        self._init_performance_optimizations()
        
        # Synchronize environment properties from servers
        self._sync_env_properties()
        
        # Initialize local buffers
        self._init_buffers()
        
        logger.info(f"EnvClient initialized with {self.num_envs} envs across {len(self.server_names)} servers")
        
    def _init_performance_optimizations(self):
        """Initialize caches and preallocated tensors for performance optimization"""
        self.server_split_indices = {}
        start_idx = 0
        for server_name in self.server_names:
            end_idx = start_idx + self.envs_per_server[server_name]
            self.server_split_indices[server_name] = (start_idx, end_idx)
            start_idx = end_idx
        
        # Preallocate commonly used tensors
        self._preallocated_tensors = {
            'zero_reward': torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            'zero_done': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            'one_reset': torch.ones(self.num_envs, device=self.device, dtype=torch.long),
            'zero_episode_length': torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        }
        
        # Thread pool for concurrent RPC calls
        self.thread_pool = ThreadPoolExecutor(max_workers=max(len(self.server_names), 8))
        
        # Precompute weights for merging info
        total_envs = sum(self.num_envs_list)
        self.info_weights = torch.tensor(self.num_envs_list, device=self.device, dtype=torch.float) / total_envs
        
        # Cache observation keys to avoid repeated lookups
        self._obs_keys_cache = None
        
    def set_episode_length_buf(self, episode_length_buf):
        """Set episode_length_buf on all servers in parallel"""
        split_episode_length_buf = self._split_tensor_for_servers(episode_length_buf)
        futures = []
        for server_name, server_episode_length_buf in split_episode_length_buf.items():
            future = self.thread_pool.submit(
                self._rpc_call_with_retry,
                server_name,
                rpc_api.set_episode_length_buf,
                (server_episode_length_buf,)
            )
            futures.append(future)
        
        success_count = 0
        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"set_episode_length_buf failed: {e}")
        
        return success_count == len(self.server_names)

    def get_episode_length_buf(self):
        """Retrieve episode_length_buf from all servers in parallel"""
        futures = []
        for server_name in self.server_names:
            future = self.thread_pool.submit(
                self._rpc_call_with_retry,
                server_name,
                rpc_api.get_episode_length_buf,
                ()
            )
            futures.append(future)

        try:
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result.to(self.device) if hasattr(result, 'to') else result)
            return torch.cat(results, dim=0) if results else torch.tensor([], device=self.device)
        except Exception as e:
            logger.error(f"get_episode_length_buf error: {e}")
            return torch.tensor([], device=self.device)
    
    def _rpc_call_with_retry(self, server_name, func, args, max_retries=2):
        """RPC call with retry mechanism"""
        for attempt in range(max_retries + 1):
            try:
                future = rpc.rpc_async(
                    server_name, 
                    func,
                    args=args,
                    timeout=self.rpc_config.get("timeout", 6000)
                )
                return future.wait()
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"RPC call to {server_name} failed after {max_retries} retries: {e}")
                    raise
                else:
                    logger.warning(f"RPC call to {server_name} failed (attempt {attempt + 1}), retrying...")
                    time.sleep(0.1 * (attempt + 1))
        
    def _calculate_envs_distribution(self) -> Dict[str, int]:
        """Determine number of environments assigned to each server"""
        distribution = {}
        if hasattr(self, 'num_envs_list') and self.num_envs_list:
            if len(self.server_names) != len(self.num_envs_list):
                logger.warning("Server count and num_envs_list length mismatch, using fallback distribution.")
                return self._fallback_distribution()
            
            total_configured_envs = sum(self.num_envs_list)
            if total_configured_envs != self.num_envs:
                logger.warning("Sum of num_envs_list doesn't match num_envs, updating num_envs accordingly.")
                self.num_envs = total_configured_envs
            
            for i, server_name in enumerate(self.server_names):
                distribution[server_name] = self.num_envs_list[i]
                
            logger.info(f"Environment distribution from config: {distribution}")
        else:
            logger.info("No num_envs_list found, using fallback distribution")
            distribution = self._fallback_distribution()
        return distribution

    def _sync_env_properties(self):
        """Synchronize environment properties from the first server"""
        try:
            first_server = self.server_names[0]
            # Example placeholder: properties = rpc.rpc_sync(first_server, rpc_api.get_env_properties, args=())
            logger.info("Environment properties synchronized")
        except Exception as e:
            logger.error(f"Failed to sync environment properties: {e}")
            raise

    def _init_buffers(self):
        """Initialize internal buffers mimicking BaseTask structure"""
        self.obs_buf_dict = {}
        self.rew_buf = self._preallocated_tensors['zero_reward'].clone()
        self.reset_buf = self._preallocated_tensors['one_reset'].clone()
        self.episode_length_buf = self._preallocated_tensors['zero_episode_length'].clone()
        self.time_out_buf = self._preallocated_tensors['zero_done'].clone()
        self.extras = {}
        self.log_dict = {}

    def _split_tensor_for_servers(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a tensor based on precomputed server indices"""
        return {name: tensor[start:end] for name, (start, end) in self.server_split_indices.items()}

    def _split_actions_for_servers(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split action tensor per server"""
        return self._split_tensor_for_servers(actions)

    def _fast_tensor_concat(self, tensors_list: List[torch.Tensor]) -> torch.Tensor:
        """Efficient tensor concatenation ensuring device consistency"""
        if not tensors_list:
            return torch.tensor([], device=self.device)
        return torch.cat([t.to(self.device, non_blocking=True) for t in tensors_list], dim=0)

    def _aggregate_server_results_optimized(self, server_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate RPC results efficiently"""
        if not server_results:
            return {}
        final_result = {}
        results_list = [server_results[name] for name in self.server_names if name in server_results]
        if not results_list:
            return final_result
        
        # Aggregate obs
        if "obs" in results_list[0]:
            final_result["obs"] = {}
            if self._obs_keys_cache is None:
                obs_keys = set()
                for result in results_list:
                    if "obs" in result:
                        obs_keys.update(result["obs"].keys())
                self._obs_keys_cache = list(obs_keys)
            
            def process_obs_key(obs_key):
                tensors = [r["obs"][obs_key] for r in results_list if "obs" in r and obs_key in r["obs"]]
                tensors = [torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t for t in tensors]
                return obs_key, self._fast_tensor_concat(tensors) if tensors else None
            
            obs_futures = {self.thread_pool.submit(process_obs_key, k): k for k in self._obs_keys_cache}
            for future in as_completed(obs_futures):
                obs_key, tensor = future.result()
                if tensor is not None:
                    final_result["obs"][obs_key] = tensor
        
        for field in ["reward", "done", "step_count"]:
            tensors = [torch.tensor(r[field], device=self.device) if not torch.is_tensor(r[field]) else r[field] 
                       for r in results_list if field in r]
            if tensors:
                final_result[field] = self._fast_tensor_concat(tensors)
        
        if any("info" in r and r["info"] for r in results_list):
            final_result["info"] = self._merge_info_optimized(results_list)
        return final_result

    def _merge_info_optimized(self, results_list: List[Dict]) -> Dict[str, Any]:
        """Merge info dicts with environment-based weighting"""
        final_info = {}
        all_info_keys = set(k for r in results_list if "info" in r and r["info"] for k in r["info"])
        
        for key in all_info_keys:
            if key in ["episode", "to_log"]:
                dicts = [r["info"][key] for r in results_list if "info" in r and key in r["info"]]
                final_info[key] = self._weighted_merge_dicts(dicts)
            else:
                values = [r["info"][key] for r in results_list if "info" in r and key in r["info"]]
                tensors = [v.to(self.device) if torch.is_tensor(v) else torch.tensor(v, device=self.device) for v in values]
                final_info[key] = torch.cat(tensors, dim=0) if all(torch.is_tensor(v) for v in tensors) else values
        return final_info

    def _weighted_merge_dicts(self, dicts: List[Dict]) -> Dict:
        """Weighted merge of dictionaries using per-server environment counts"""
        if not dicts:
            return {}
        if len(dicts) == 1:
            return dicts[0]
        result = {}
        weights = self.info_weights[:len(dicts)]
        weights /= weights.sum()
        
        for key in {k for d in dicts for k in d}:
            values, valid_weights = [], []
            for i, d in enumerate(dicts):
                if key in d:
                    v = d[key]
                    if not torch.is_tensor(v):
                        v = torch.tensor(v, device=self.device)
                    values.append(v.to(self.device))
                    valid_weights.append(weights[i])
            if values:
                valid_weights = torch.stack(valid_weights)
                valid_weights /= valid_weights.sum()
                result[key] = (torch.stack(values) * valid_weights.unsqueeze(-1)).sum(dim=0)
        return result

    def step(self, actor_state: Dict[str, Any]) -> tuple:
        """Perform one parallel environment step across all servers"""
        actions = actor_state["actions"]
        split_actions = self._split_actions_for_servers(actions)
        
        futures, server_results = [], {}
        for name, act in split_actions.items():
            future = self.thread_pool.submit(self._rpc_call_with_retry, name, rpc_api.step, ({"actions": act},))
            futures.append((name, future))
        
        for name, future in futures:
            try:
                server_results[name] = future.result()
            except Exception as e:
                logger.error(f"Step RPC call to {name} failed: {e}")
                raise

        aggregated = self._aggregate_server_results_optimized(server_results)
        if "obs" in aggregated:
            self.obs_buf_dict = aggregated["obs"]
        if "reward" in aggregated:
            self.rew_buf = aggregated["reward"]
        if "done" in aggregated:
            self.reset_buf = aggregated["done"].long()

        return (
            aggregated.get("obs", {}),
            aggregated.get("reward", self._preallocated_tensors['zero_reward'].clone()),
            aggregated.get("done", self._preallocated_tensors['zero_done'].clone()),
            aggregated.get("info", {})
        )

    def reset_all(self) -> Dict[str, Any]:
        """Reset all environments on all servers"""
        futures = [(name, self.thread_pool.submit(self._rpc_call_with_retry, name, rpc_api.reset_all, ())) 
                   for name in self.server_names]
        results = []
        for name, future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Reset RPC call to {name} failed: {e}")
                raise

        final_obs = {}
        if results:
            all_keys = {k for r in results if isinstance(r, dict) for k in r.keys()}
            def process_key(k):
                tensors = [torch.tensor(r[k], device=self.device) if not torch.is_tensor(r[k]) else r[k]
                           for r in results if isinstance(r, dict) and k in r]
                return k, self._fast_tensor_concat(tensors)
            futures = {self.thread_pool.submit(process_key, k): k for k in all_keys}
            for f in as_completed(futures):
                k, t = f.result()
                final_obs[k] = t

        self.reset_buf = self._preallocated_tensors['one_reset'].clone()
        self.episode_length_buf = self._preallocated_tensors['zero_episode_length'].clone()
        logger.info(f"Reset completed, obs keys: {list(final_obs.keys())}")
        return final_obs

    def set_is_evaluating(self) -> bool:
        """Set evaluation mode on all servers"""
        futures = [self.thread_pool.submit(self._rpc_call_with_retry, s, rpc_api.set_is_evaluating, ()) 
                   for s in self.server_names]
        success = 0
        for f in as_completed(futures):
            try:
                if f.result():
                    success += 1
            except Exception as e:
                logger.error(f"set_is_evaluating failed: {e}")
        return success == len(self.server_names)

    def render(self, sync_frame_time: bool = True):
        """Render environment on the first server only"""
        if self.server_names:
            try:
                rpc.rpc_sync(
                    self.server_names[0],
                    "render",
                    args=(sync_frame_time,),
                    timeout=self.rpc_config.get("timeout", 6000)
                )
            except Exception as e:
                logger.warning(f"Render call failed: {e}")

    def close(self):
        """Gracefully close all RPC connections and threads"""
        try:
            self.thread_pool.shutdown(wait=True)
            for s in self.server_names:
                try:
                    rpc.rpc_sync(s, "cleanup", args=(), timeout=self.rpc_config.get("timeout", 6000))
                except Exception as e:
                    logger.warning(f"Cleanup call to {s} failed: {e}")
            rpc.shutdown()
            logger.info("RPC shutdown completed")
        except Exception as e:
            logger.error(f"Failed to shutdown RPC: {e}")

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def merge_infos_weighted(self, infos, num_envs_list, device="cuda:0"):
        """Deprecated: use internal optimized version instead"""
        logger.warning("Using deprecated merge_infos_weighted; please switch to optimized version.")
        return self._merge_info_optimized([{"info": infos}])
