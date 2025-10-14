import json
import os
import re
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

import threading
# from pynput import keyboard

def on_press(key, env):
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
        # Force Control
        if hasattr(key, 'char'):
            if key.char == '1':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        pass

def listen_for_keypress(env):
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()

def aggregate_metrics(metrics_list):
    agg = {}
    if not metrics_list:
        return agg
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list]
        agg[k] = float(np.mean(vals))
    return agg
# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg

@hydra.main(config_path="config", config_name="base_eval_multi")
def main(override_config: OmegaConf):
    # ------------------- Logging Setup -------------------
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())
    
    # ------------------- Config Loading -------------------
    if override_config.checkpointdir is not None:
        has_config = True
        checkpoint = Path(override_config.checkpointdir)
        config_path = checkpoint / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    # ------------------- Simulator Initialization -------------------       
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        # device = config.device
        device = "cuda:0"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ------------------- Evaluation Log Setup -------------------
    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    config.env.config.save_rendering_dir = str(checkpoint / "renderings" / f"ckpt")
    config.env.config.ckpt_dir = str(checkpoint) # commented out for now, might need it back to save motion
    env = instantiate(config.env, device=device)

    results_all = {}
    parent_dir = os.path.dirname(config.checkpointdir)
    basename = os.path.basename(config.checkpointdir)
    save_path = f"{parent_dir}/{simulator_type}_{config.checkpointdir.split('/')[-1]}.jsonl"
    print(f"Save eval results to {save_path}")

    # ------------------- Checkpoint Sorting -------------------
    cpkt_list = [f for f in os.listdir(checkpoint) if f.endswith('.pt')]
    def extract_last_number(filename):
        numbers = re.findall(r'(\d+)', filename)
        return int(numbers[-1]) if numbers else -1
    cpkt_list = sorted(cpkt_list, key=extract_last_number, reverse=True)
    cpkt_list = cpkt_list[::4][:5]  # Sample a subset of checkpoints
    with open(save_path, "w") as f:
        pass  # Clear existing file

    # ------------------- Helper: JSON Serialization -------------------
    def make_json_serializable(obj):
        """
        Convert an object (dict/list/scalar/array/tensor) into JSON-serializable types.
        """
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return make_json_serializable(obj.tolist())
        elif isinstance(obj, (np.float32, np.float64, torch.float32, torch.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, torch.int32, torch.int64)):
            return int(obj)
        else:
            return obj
    
    # ------------------- Evaluation Loop -------------------
    for cpkt in cpkt_list:
        print(f"Evaluate {str(checkpoint / cpkt)}")
        env.reset_all()

        # Determine number of evaluations per checkpoint
        num_eval_runs = 10 if simulator_type == 'MuJoCo' else 1
        all_results = []
        all_metrics_list = []
        success_metrics_list = []

        for _ in range(num_eval_runs):
            algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
            algo.setup()
            algo.load(str(checkpoint / cpkt))

            EXPORT_POLICY = False
            EXPORT_ONNX = True

            checkpoint_path = str(checkpoint / cpkt)
            checkpoint_dir = os.path.dirname(checkpoint_path)
            HV_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            exported_policy_path = os.path.join(HV_ROOT_DIR, checkpoint_dir, 'exported')
            os.makedirs(exported_policy_path, exist_ok=True)
            exported_policy_name = checkpoint_path.split('/')[-1]
            exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

            # Optional policy export
            if EXPORT_POLICY:
                export_policy_as_jit(algo.alg.actor_critic, exported_policy_path, exported_policy_name)
                logger.info('Exported policy as jit script to: ', os.path.join(exported_policy_path, exported_policy_name))
            if EXPORT_ONNX:
                example_obs_dict = algo.get_example_obs()
                export_policy_as_onnx(algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)
                logger.info(f'Exported policy as onnx to: {os.path.join(exported_policy_path, exported_onnx_name)}')

            # Evaluate policy
            results = algo.evaluate_policy()
            all_results.append(results)
            all_metrics_list.append(results["metrics_all"])
            if results.get("success_rate", 0.0) > 0:
                success_metrics_list.append(results.get("metrics_succ", {}))

        # Aggregate metrics for MuJoCo (multiple runs) or single run for others
        if simulator_type == 'MuJoCo':
            avg_all_metrics = aggregate_metrics(all_metrics_list)
            avg_success_metrics = aggregate_metrics(success_metrics_list)
            avg_success_rate = float(np.mean([r["success_rate"] for r in all_results]))
            final_result = {
                "success_rate": avg_success_rate,
                "metrics_all": avg_all_metrics,
                "metrics_succ": avg_success_metrics
            }
            final_result = make_json_serializable(final_result)
        else:
            final_result = make_json_serializable(all_results[0])

        results_all[cpkt] = final_result

        # Save results to JSONL
        with open(save_path, "a") as f:
            f.write(json.dumps({cpkt: final_result}) + "\n")
    return 0


if __name__ == "__main__":
    main()
