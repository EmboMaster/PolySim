# ======================================================================================
#
# polysim_train_agent.py
#
# Description:
# This script serves as the main entry point for a distributed reinforcement learning
# training framework. It orchestrates the entire process by:
#   1. Launching multiple heterogeneous simulation environments (e.g., Isaac Gym, 
#      Isaac Sim, Genesis) as background server processes using Torch RPC.
#   2. Initializing a single client (`EnvClient`) that communicates with all
#      simulation servers, presenting them as a unified, vectorized environment.
#   3. Instantiating and running the reinforcement learning algorithm which uses
#      the unified client for training.
#   4. Ensuring robust cleanup of all background server processes upon completion
#      or interruption.
#
# Usage:
#   This script is intended to be run using Hydra for configuration management.
#   Example: python polysim_train_agent.py +simulator_list=[isaacgym,isaacsim] ...
#
# ======================================================================================

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from loguru import logger
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

from env_client import EnvClient

# Global list to keep track of server subprocesses for cleanup
server_processes = []

def create_env_client(master_port, simulator_lists, rank, world_size, num_envs_list, device_list):
    """
    Configures and initializes the RPC client (`EnvClient`).

    This function sets up the necessary RPC configurations, environment variables,
    and initializes the connection from this client process to all server processes.

    Args:
        master_port (int): The port for the RPC master process.
        simulator_lists (list): A list of simulator names (e.g., ['isaacgym', 'isaacsim']).
        rank (int): The rank of this client process in the RPC world.
        world_size (int): The total number of processes (servers + 1 client).
        num_envs_list (list): A list of the number of environments per simulator.
        device_list (list): A list of CUDA devices for all processes.

    Returns:
        EnvClient: An initialized instance of the environment client.
    """
    # Load the base configuration file for the client
    config = OmegaConf.load("humanoidverse/config/base_client.yaml")

    # Dynamically update the configuration based on runtime arguments
    config.num_envs = sum(num_envs_list)
    config.num_envs_list = num_envs_list
    config.rpc.server_names = simulator_lists
    config.rpc.master_port = master_port
    config.rpc.rank = rank
    config.rpc.world_size = world_size
    config.rpc.device = device_list[rank]
    config.rpc.client_gpu = int(device_list[rank])
    config.rpc.server_gpu_list = [int(device_list[i]) for i in range(len(simulator_lists))]

    # Set environment variables required by PyTorch Distributed and RPC
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo") # Use loopback interface for local communication
    os.environ.setdefault("TP_SOCKET_IFNAME", "lo")

    # Configure RPC options for the TensorPipe backend
    opts = TensorPipeRpcBackendOptions(num_worker_threads=64, rpc_timeout=6000)
    
    # Define the GPU device mapping between the client and each server.
    # This enables direct GPU-to-GPU data transfers, which is crucial for performance.
    for i, simulator_name in enumerate(simulator_lists):
        client_gpu = int(device_list[rank])
        server_gpu = int(device_list[i])
        opts.set_device_map(simulator_name, {client_gpu: server_gpu})
        logger.info(f"RPC device map: client:{client_gpu} -> server '{simulator_name}':{server_gpu}")

    # Initialize the RPC framework for this client process
    logger.info("[Client] Initializing RPC...")
    rpc.init_rpc(name="ppo_trainer", rank=rank, world_size=world_size, rpc_backend_options=opts)
    logger.info("[Client] RPC initialized successfully.")

    try:
        # Create and return the EnvClient instance
        logger.info(f"[Client] Creating EnvClient on device {device_list[rank]}...")
        env_client = EnvClient(config, device=device_list[rank])
        logger.info("[Client] EnvClient created successfully.")
        return env_client
    except Exception as e:
        # If client creation fails, log the error and perform a clean shutdown.
        logger.error(f"[Client] Failed to create EnvClient: {e}")
        import traceback
        traceback.print_exc()
        rpc.shutdown()
        logger.error("[Client] RPC shut down due to an error.")
        sys.exit(1)

def cleanup_servers(sig=None, frame=None):
    """
    Terminates all running server subprocesses.

    This function is registered as a signal handler and is also called in the
    `finally` block to ensure that no zombie processes are left behind.
    """
    logger.info("Cleaning up server processes...")
    for p in server_processes:
        if p.poll() is None:  # Check if the process is still running
            # Use os.killpg to terminate the entire process group, ensuring all children die
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            logger.warning(f"Process group for PID {p.pid} has been killed.")
    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)

def parse_common_args():
    """
    Parses command-line arguments, filtering out those specific to this launcher script.

    The remaining "common" arguments (e.g., algorithm hyperparameters) are passed
    directly to the server subprocesses to ensure consistent configuration.

    Returns:
        list: A list of common command-line arguments.
    """
    common_args = []
    for arg in sys.argv[1:]:
        # These arguments are for the launcher only, not for the servers
        if not (arg.startswith("+simulator_list") or arg.startswith("num_envs_list") or 
                arg.startswith("master_port") or arg.startswith("device_list")):
            common_args.append(arg)
    return common_args

@hydra.main(config_path="config", config_name="base_overall", version_base="1.1")
def main(config: DictConfig):
    """
    Main function to orchestrate the distributed training process.
    """
    # Register signal handlers to ensure `cleanup_servers` is called on interruption (Ctrl+C).
    signal.signal(signal.SIGINT, cleanup_servers)
    signal.signal(signal.SIGTERM, cleanup_servers)

    # ======================================================================================
    # 1. Launch Simulation Servers
    # ======================================================================================
    logger.info(">>> STEP 1: Launching simulation servers in the background...")
    
    # Extract server configurations from the Hydra config object
    server_configs = config.simulator_list
    num_envs_list = config.num_envs_list
    world_size = len(server_configs) + 1  # Number of servers + 1 client
    master_port = config.master_port
    device_list = config.device_list
    common_args = parse_common_args()

    # Create a timestamped directory to store server logs for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("server_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = []

    try:
        # Iterate through the server configurations and launch each one as a subprocess
        for i, server_name in enumerate(server_configs):
            # Select the correct conda environment for each simulator
            conda_env_map = {'isaacgym': 'hvgym', 'isaacsim': 'hvlab', 'genesis': 'hvgen'}
            if server_name not in conda_env_map:
                raise ValueError(f"Simulator '{server_name}' is not supported.")
            conda_env = conda_env_map[server_name]

            # Construct the command to launch the server script
            cmd_parts = [
                f"conda run -n {conda_env}",
                "python -u humanoidverse/hydra_server.py", # -u for unbuffered output
                f"+rpc={server_name}",
                f"+simulator={server_name}",
                f"device=cuda:{device_list[i]}",
                f"num_envs={num_envs_list[i]}",
                f"rpc.name={server_name}",
                f"rpc.rank={i}",
                f"rpc.world_size={world_size}",
                f"rpc.master_port={master_port}",
                f"rpc.server_gpu={int(device_list[i])}",
                f"rpc.client_gpu={int(device_list[-1])}", # Client is the last process
            ]
            cmd_parts.extend(common_args)
            
            # Set up simulator-specific environment variables and command prefixes
            if server_name == 'isaacgym':
                # Isaac Gym requires its library path to be explicitly set.
                prefix = f"export LD_LIBRARY_PATH=/home/embodied/miniconda3/envs/{conda_env}/lib:$LD_LIBRARY_PATH &&"
            elif server_name == 'isaacsim':
                # Isaac Sim requires unsetting several paths to avoid conflicts.
                prefix = "unset PYTHONPATH && unset LD_LIBRARY_PATH &&"
            else:
                prefix = ""
            
            final_cmd = f"{prefix} export CUDA_LAUNCH_BLOCKING=1 && {' '.join(cmd_parts)}"

            # Redirect server stdout/stderr to a dedicated log file
            log_path = log_dir / f"{server_name}_rank{i}.log"
            logger.info(f"Redirecting output of [{server_name}] to {log_path}")
            log_file = open(log_path, 'w')
            log_files.append(log_file)

            logger.info(f"Launching [{server_name}] with command:\n{final_cmd}\n")
            
            # Launch the server process in the background
            proc = subprocess.Popen(
                final_cmd, 
                shell=True, 
                preexec_fn=os.setsid, # Creates a new process group for easy cleanup
                stdout=log_file,
                stderr=subprocess.STDOUT, # Redirect stderr to the same log file
                env={**os.environ, "PYTHONUNBUFFERED": "1"} # Ensure output is not buffered
            )
            server_processes.append(proc)

        # Health check: Wait briefly and check if any server processes have crashed
        logger.info(f"Launched {len(server_processes)} servers. Verifying their status...")
        time.sleep(70) # Wait for servers to initialize
        
        for i, p in enumerate(server_processes):
            exit_code = p.poll()
            if exit_code is not None:
                server_name = server_configs[i]
                logger.error(f"!!! Server [{server_name}] (PID: {p.pid}) terminated prematurely with exit code {exit_code}.")
                logger.error(f"Check its log file for details: {log_files[i].name}")
                raise RuntimeError(f"Server {server_name} failed to start.")

        # ======================================================================================
        # 2. Initialize the Unified Environment Client
        # ======================================================================================
        logger.info(">>> STEP 2: Initializing unified EnvClient...")
        
        env = create_env_client(
            master_port=master_port,
            simulator_lists=server_configs,
            rank=world_size - 1, # Client has the highest rank
            world_size=world_size,
            num_envs_list=num_envs_list,
            device_list=device_list
        )

        # ======================================================================================
        # 3. Start the Training Process
        # ======================================================================================
        logger.info(">>> STEP 3: Starting training process...")
        
        device = env.device
        experiment_save_dir = Path(config.experiment_dir)
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the exact configuration used for this run for reproducibility
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(config, file)

        # Instantiate the learning algorithm using the Hydra config
        algo = hydra.utils.instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)
        algo.setup()

        # Load a pre-trained model checkpoint if specified in the config
        if config.get("checkpoint") is not None:
            logger.info(f"Loading checkpoint from: {config.checkpoint}")
            algo.load(config.checkpoint)

        # Start the main training loop
        algo.learn()
        
        logger.info("Training finished.")

    finally:
        # ======================================================================================
        # 4. Cleanup
        # ======================================================================================
        logger.info(">>> STEP 4: Cleaning up...")
        if 'env' in locals():
            env.close()  # Gracefully close the client and its RPC connections
        
        # Close all log files
        for f in log_files:
            f.close()
            
        cleanup_servers() # Terminate all server subprocesses

if __name__ == "__main__":
    main()