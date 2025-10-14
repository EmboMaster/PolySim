#!/usr/bin/env python3
"""
ManiSkill训练示例脚本
展示如何使用ManiSkill仿真器进行机器人操作任务训练
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from humanoidverse.simulator.maniskill.maniskill import ManiSkill
from humanoidverse.simulator.maniskill.maniskill_cfg import ManiSkillCfg
from omegaconf import OmegaConf


def create_maniskill_config(task_name="PickCube-v1", robot_type="panda", num_envs=4):
    """创建ManiSkill配置"""
    
    config = OmegaConf.create({
        'simulator': {
            '_target_': 'humanoidverse.simulator.maniskill.maniskill.ManiSkill',
            'config': {
                'name': 'maniskill',
                'sim': {
                    'fps': 200,
                    'control_decimation': 4,
                    'substeps': 1,
                    'gravity': [0.0, 0.0, -9.81],
                    'solver_iterations': 4,
                    'solver_velocity_iterations': 1,
                    'render_mode': 'human',
                    'render_interval': 4
                },
                'scene': {
                    'num_envs': num_envs,
                    'env_spacing': 4.0,
                    'replicate_physics': True
                },
                'task': {
                    'task_name': task_name,
                    'control_mode': 'pd_ee_delta_pose',
                    'obs_mode': 'state',
                    'reward_mode': 'dense'
                },
                'robot': {
                    'robot_type': robot_type,
                    'asset_path': None
                },
                'env': {
                    'max_episode_steps': 1000,
                    'action_scale': 1.0,
                    'reward_scale': 1.0
                }
            }
        },
        'robot': {
            'asset': {
                'robot_type': robot_type,
                'asset_path': None
            }
        },
        'terrain': {
            'mesh_type': 'plane'
        },
        'num_envs': num_envs,
        'headless': False,
        'domain_rand': {
            'randomize_friction': True,
            'friction_range': [0.5, 1.5],
            'randomize_mass': True,
            'mass_range': [0.8, 1.2],
            'randomize_com': True,
            'com_range': [-0.1, 0.1]
        }
    })
    
    return config


def train_maniskill_agent(config, num_episodes=1000):
    """训练ManiSkill智能体"""
    
    print("开始ManiSkill智能体训练...")
    
    # 创建仿真器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    simulator = ManiSkill(config, device)
    simulator.set_headless(False)
    
    try:
        # 设置仿真器
        simulator.setup()
        print("✓ 仿真器设置完成")
        
        # 设置地形
        simulator.setup_terrain('plane')
        print("✓ 地形设置完成")
        
        # 加载机器人资产
        dof_info = simulator.load_assets(config.robot)
        if dof_info:
            num_dof, num_bodies, dof_names, body_names = dof_info
            print(f"✓ 机器人加载成功: {num_dof} DOFs, {num_bodies} bodies")
        else:
            print("⚠ 机器人加载失败，使用默认值")
            simulator.num_dof = 7  # Panda机器人默认7个关节
            simulator.num_bodies = 8  # 包括基座
        
        # 创建环境
        num_envs = config.num_envs
        env_origins = torch.zeros(num_envs, 3)
        base_init_state = torch.zeros(7 + simulator.num_dof)  # 位置(3) + 四元数(4) + 关节位置
        
        envs, handles = simulator.create_envs(
            num_envs, env_origins, base_init_state, config
        )
        print(f"✓ 环境创建成功: {len(envs)} 个环境")
        
        # 准备仿真
        simulator.prepare_sim()
        print("✓ 仿真准备完成")
        
        # 设置查看器
        if not config.headless:
            simulator.setup_viewer()
            print("✓ 查看器设置完成")
        
        # 训练循环
        print(f"\n开始训练，共 {num_episodes} 个episode...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # 重置环境
            simulator.reset_envs(torch.arange(num_envs, device=device))
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                # 随机动作（这里可以替换为实际的策略网络）
                actions = torch.randn(num_envs, simulator.num_dof) * 0.1
                
                # 执行仿真步骤
                simulator.step(actions)
                
                # 获取观察和奖励
                observations = simulator.get_observations()
                
                # 计算奖励（这里需要根据具体任务实现）
                rewards = torch.ones(num_envs, device=device) * 0.1
                episode_reward += rewards.mean().item()
                
                # 检查是否完成
                # 这里需要根据具体任务实现终止条件
                if episode_length > 500:  # 简单的终止条件
                    done = True
                
                episode_length += 1
                
                # 渲染
                if not config.headless:
                    simulator.render()
            
            # 记录episode信息
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Length: {avg_length:.1f}")
        
        # 训练完成
        print(f"\n🎉 训练完成！")
        print(f"最终平均奖励: {np.mean(episode_rewards):.3f}")
        print(f"最终平均长度: {np.mean(episode_lengths):.1f}")
        
        return episode_rewards, episode_lengths
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        # 关闭仿真器
        simulator.close()
        print("✓ 仿真器已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ManiSkill训练示例")
    parser.add_argument("--task", type=str, default="PickCube-v1", 
                       help="任务名称")
    parser.add_argument("--robot", type=str, default="panda", 
                       help="机器人类型")
    parser.add_argument("--num_envs", type=int, default=4, 
                       help="环境数量")
    parser.add_argument("--num_episodes", type=int, default=100, 
                       help="训练episode数量")
    parser.add_argument("--headless", action="store_true", 
                       help="无头模式")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ManiSkill训练示例")
    print("=" * 60)
    print(f"任务: {args.task}")
    print(f"机器人: {args.robot}")
    print(f"环境数量: {args.num_envs}")
    print(f"训练episode: {args.num_episodes}")
    print(f"无头模式: {args.headless}")
    print("=" * 60)
    
    # 创建配置
    config = create_maniskill_config(
        task_name=args.task,
        robot_type=args.robot,
        num_envs=args.num_envs
    )
    config.headless = args.headless
    
    # 开始训练
    rewards, lengths = train_maniskill_agent(config, args.num_episodes)
    
    if rewards is not None:
        print("\n训练结果:")
        print(f"总episode: {len(rewards)}")
        print(f"平均奖励: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"最佳奖励: {np.max(rewards):.3f}")
        print(f"最差奖励: {np.min(rewards):.3f}")


if __name__ == "__main__":
    main()
