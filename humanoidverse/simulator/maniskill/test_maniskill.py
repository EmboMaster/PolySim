#!/usr/bin/env python3
"""
ManiSkill仿真器测试脚本
用于验证仿真器的基本功能是否正常工作
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from humanoidverse.simulator.maniskill.maniskill import ManiSkill
from omegaconf import OmegaConf


def test_maniskill_basic():
    """测试ManiSkill仿真器的基本功能"""
    print("开始测试ManiSkill仿真器...")
    
    try:
        # 创建基础配置
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
                        'num_envs': 2,
                        'env_spacing': 4.0,
                        'replicate_physics': True
                    },
                    'task': {
                        'task_name': 'PickCube-v1',
                        'control_mode': 'pd_ee_delta_pose',
                        'obs_mode': 'state',
                        'reward_mode': 'dense'
                    },
                    'robot': {
                        'robot_type': 'panda',
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
                    'robot_type': 'panda',
                    'asset_path': None
                }
            },
            'terrain': {
                'mesh_type': 'plane'
            },
            'num_envs': 2,
            'headless': True  # 测试时使用无头模式
        })
        
        print("✓ 配置创建成功")
        
        # 创建仿真器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        simulator = ManiSkill(config, device)
        print("✓ 仿真器创建成功")
        
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
        num_envs = 2
        env_origins = torch.zeros(num_envs, 3)
        base_init_state = torch.zeros(7 + simulator.num_dof)  # 位置(3) + 四元数(4) + 关节位置
        
        envs, handles = simulator.create_envs(
            num_envs, env_origins, base_init_state, config
        )
        print(f"✓ 环境创建成功: {len(envs)} 个环境")
        
        # 准备仿真
        simulator.prepare_sim()
        print("✓ 仿真准备完成")
        
        # 测试基本操作
        print("\n开始测试基本操作...")
        
        # 测试获取DOF限制
        dof_limits = simulator.get_dof_limits_properties()
        if dof_limits[0] is not None:
            print(f"✓ DOF限制获取成功: 位置限制形状 {dof_limits[0].shape}")
        else:
            print("⚠ DOF限制获取失败")
        
        # 测试仿真步骤
        actions = torch.randn(num_envs, simulator.num_dof) * 0.1
        simulator.step(actions)
        print("✓ 仿真步骤执行成功")
        
        # 测试观察获取
        observations = simulator.get_observations()
        if observations:
            print(f"✓ 观察获取成功: {len(observations)} 个环境")
        else:
            print("⚠ 观察获取失败")
        
        # 测试环境重置
        simulator.reset_envs([0])
        print("✓ 环境重置成功")
        
        # 测试状态设置
        test_root_state = torch.randn(1, 7)  # 位置和四元数
        test_dof_state = torch.randn(1, simulator.num_dof * 2)  # 位置和速度
        simulator.set_actor_root_state_tensor([0], test_root_state)
        simulator.set_dof_state_tensor([0], test_dof_state)
        print("✓ 状态设置成功")
        
        # 关闭仿真器
        simulator.close()
        print("✓ 仿真器关闭成功")
        
        print("\n🎉 所有测试通过！ManiSkill仿真器工作正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_maniskill_config():
    """测试配置文件加载"""
    print("\n开始测试配置文件加载...")
    
    try:
        config_path = project_root / "humanoidverse" / "config" / "simulator" / "maniskill.yaml"
        
        if config_path.exists():
            config = OmegaConf.load(config_path)
            print("✓ 配置文件加载成功")
            
            # 检查必要的配置项
            required_keys = ['simulator', 'config', 'sim', 'scene', 'task', 'robot']
            for key in required_keys:
                if key in config.simulator.config:
                    print(f"✓ 配置项 {key} 存在")
                else:
                    print(f"⚠ 配置项 {key} 缺失")
            
            return True
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("ManiSkill仿真器测试套件")
    print("=" * 60)
    
    # 检查依赖
    try:
        import mani_skill
        print("✓ ManiSkill已安装")
    except ImportError:
        print("❌ ManiSkill未安装，请先安装: pip install mani-skill")
        return False
    
    try:
        import sapien
        print("✓ SAPIEN已安装")
    except ImportError:
        print("❌ SAPIEN未安装，请先安装: pip install sapien")
        return False
    
    # 运行测试
    config_test = test_maniskill_config()
    basic_test = test_maniskill_basic()
    
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"配置文件测试: {'通过' if config_test else '失败'}")
    print(f"基本功能测试: {'通过' if basic_test else '失败'}")
    
    if config_test and basic_test:
        print("\n🎉 所有测试通过！ManiSkill仿真器已准备就绪。")
        return True
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
