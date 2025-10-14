# server_hydra.py
from __future__ import annotations
import os, sys, signal, threading, logging
from pathlib import Path
from typing import Dict, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from loguru import logger
# from hydra.utils import instantiate

import rpc_api  

from utils.config_utils import *  # noqa: F403
# from humanoidverse.utils.helpers import pre_process_config

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

class EnvService:
    def __init__(self, env, device: str):
        self.env = env
        self.device = device
        self._n = 0

    def step(self, d: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        self._n += 1
        res = self.env.step(d)  # 期望返回 (obs_dict, reward, done, info)

        if isinstance(res, tuple) and len(res) == 4:
            obs, rew, done, info = res
            dev = rew.device if hasattr(rew, "device") else self.device
            print("rew: ",rew[:10])
            return {
                "obs":  obs,
                "reward": rew,
                "done": done,
                "info": info,
                "step_count": torch.tensor([self._n], device=dev, dtype=torch.long),
            }
        return {"out": res}
    def reset_all(self) -> Dict[str, Any]:
        res = self.env.reset_all()
        return res if isinstance(res, dict) else {"obs": res}
    def  set_is_evaluating(self) -> bool:
        try:
            self.env.set_is_evaluating()  
            return True
        except Exception as e:
            return False
    def set_episode_length_buf(self,episode_length_buf) -> bool:
        try:
            self.env.episode_length_buf = episode_length_buf
            return True
        except Exception as e:
            return False
    
    def get_episode_length_buf(self):
        try:
            episode_length_buf = self.env.episode_length_buf
            return episode_length_buf
        except Exception as e:
            return []
def wait_for_termination():
    e = threading.Event()
    def _stop(sig, frm): e.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    e.wait()

@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacGym':
        import isaacgym  
    # import ipdb; ipdb.set_trace()
    # from utils.config_utils import *  # noqa: F403
    from humanoidverse.utils.helpers import pre_process_config
    from hydra.utils import instantiate

    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
        # import ipdb; ipdb.set_trace()# noqa: F401


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(project=project_name, 
                entity=config.wandb.wandb_entity,
                name=run_name,
                sync_tensorboard=True,
                config=unresolved_conf,
                dir=wandb_dir)
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)

    # torch.set_float32_matmul_precision("medium")

    # fabric: Fabric = instantiate(config.fabric)
    # fabric.launch()

    # if config.seed is not None:
    #     rank = fabric.global_rank
    #     if rank is None:
    #         rank = 0
    #     fabric.seed_everything(config.seed + rank)
    #     seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env: BaseEnv = instantiate(config=config.env, device=device)
    
    print("Env Device:", device)
    
    service = EnvService(env, device=device)
    rpc_api.register_service(service)  
    print(f"[server] Env initialized on {device} and registered.")

    import torch.distributed.rpc as rpc
    from torch.distributed.rpc import TensorPipeRpcBackendOptions

    os.environ["MASTER_ADDR"] = str(config.rpc.master_addr)      # 127.0.0.1
    os.environ["MASTER_PORT"] = str(config.rpc.master_port)      # "29550"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("TP_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_USE_IPV6", "0")

    opts = TensorPipeRpcBackendOptions(num_worker_threads=64, rpc_timeout=6000)
    client_name = str(getattr(config.rpc, "client_name", "ppo_trainer"))

    print(f"{int(config.rpc.server_gpu)}, {int(config.rpc.client_gpu)}")
    opts.set_device_map(client_name, {int(config.rpc.server_gpu): int(config.rpc.client_gpu)})

    print("before init:", config.rpc.name, config.rpc.rank, config.rpc.world_size,
          "|", os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    rpc.init_rpc(
        name=str(config.rpc.name),                # "isaacsim"
        rank=int(config.rpc.rank),                # 0
        world_size=int(config.rpc.world_size),    # 2
        rpc_backend_options=opts,
    )
    print("after init")
    print(f"[{config.rpc.name}] Ready. Client can call rpc_api.step(d).")
    try:
        wait_for_termination()
    finally:
        try:
            rpc.shutdown()
        except Exception as e:
            print(f"[{config.rpc.name}] rpc.shutdown failed:", e)
        print(f"[{config.rpc.name}] RPC shutdown.")

if __name__ == "__main__":
    main()