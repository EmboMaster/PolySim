#!/usr/bin/env python3
"""
测试ManiSkill G1训练配置
验证所有配置文件是否正确设置
"""

import sys
import os
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_config_files():
    """测试配置文件是否存在和正确"""
    print("🔍 测试配置文件...")
    
    # 测试仿真器配置
    simulator_config_path = project_root / "humanoidverse" / "config" / "simulator" / "maniskill_g1.yaml"
    if simulator_config_path.exists():
        print("✅ 仿真器配置文件存在: maniskill_g1.yaml")
        
        # 读取并验证配置
        with open(simulator_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 检查关键配置项
        if 'simulator' in config:
            sim_config = config['simulator']['config']
            print(f"   - 仿真器名称: {sim_config.get('name', 'N/A')}")
            print(f"   - 机器人类型: {sim_config.get('robot', {}).get('robot_type', 'N/A')}")
            print(f"   - 任务名称: {sim_config.get('task', {}).get('task_name', 'N/A')}")
            print(f"   - 控制模式: {sim_config.get('task', {}).get('control_mode', 'N/A')}")
            
            # 检查G1配置
            if 'g1_config' in sim_config:
                g1_config = sim_config['g1_config']
                print(f"   - G1 DOF数量: {g1_config.get('num_dof', 'N/A')}")
                print(f"   - G1关节数量: {len(g1_config.get('joint_names', []))}")
                print(f"   - G1身体部位数量: {len(g1_config.get('body_names', []))}")
            else:
                print("   ❌ 缺少G1配置")
    else:
        print("❌ 仿真器配置文件不存在: maniskill_g1.yaml")
        return False
    
    # 测试环境配置
    env_config_path = project_root / "humanoidverse" / "config" / "env" / "motion_tracking_maniskill.yaml"
    if env_config_path.exists():
        print("✅ 环境配置文件存在: motion_tracking_maniskill.yaml")
        
        # 读取并验证配置
        with open(env_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 检查关键配置项
        if 'env' in config:
            env_config = config['env']
            print(f"   - 环境目标类: {env_config.get('_target_', 'N/A')}")
            
            # 检查ManiSkill配置
            if 'config' in env_config and 'maniskill' in env_config['config']:
                maniskill_config = env_config['config']['maniskill']
                print(f"   - 任务配置: {maniskill_config.get('task_config', {})}")
                print(f"   - G1配置: {maniskill_config.get('g1_config', {})}")
            else:
                print("   ❌ 缺少ManiSkill配置")
    else:
        print("❌ 环境配置文件不存在: motion_tracking_maniskill.yaml")
        return False
    
    # 测试实验配置
    exp_config_path = project_root / "humanoidverse" / "config" / "exp" / "motion_tracking_maniskill.yaml"
    if exp_config_path.exists():
        print("✅ 实验配置文件存在: motion_tracking_maniskill.yaml")
        
        # 读取并验证配置
        with open(exp_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 检查关键配置项
        if 'exp' in config:
            exp_config = config['exp']
            print(f"   - 实验名称: {exp_config.get('name', 'N/A')}")
            print(f"   - 仿真器: {exp_config.get('simulator', 'N/A')}")
            print(f"   - 环境: {exp_config.get('env', 'N/A')}")
            print(f"   - 机器人: {exp_config.get('robot', 'N/A')}")
    else:
        print("❌ 实验配置文件不存在: motion_tracking_maniskill.yaml")
        return False
    
    return True

def test_python_modules():
    """测试Python模块是否存在"""
    print("\n🔍 测试Python模块...")
    
    # 测试ManiSkill仿真器
    try:
        from humanoidverse.simulator.maniskill.maniskill import ManiSkill
        print("✅ ManiSkill仿真器模块导入成功")
    except ImportError as e:
        print(f"❌ ManiSkill仿真器模块导入失败: {e}")
        return False
    
    # 测试ManiSkill环境
    try:
        from humanoidverse.envs.motion_tracking.maniskill_motion_tracking import ManiSkillG1MotionTracking
        print("✅ ManiSkill运动跟踪环境模块导入成功")
    except ImportError as e:
        print(f"❌ ManiSkill运动跟踪环境模块导入失败: {e}")
        return False
    
    # 测试配置模块
    try:
        from humanoidverse.simulator.maniskill.maniskill_cfg import MANISKILL_CFG
        print("✅ ManiSkill配置模块导入成功")
    except ImportError as e:
        print(f"❌ ManiSkill配置模块导入失败: {e}")
        return False
    
    return True

def test_dependencies():
    """测试依赖包是否安装"""
    print("\n🔍 测试依赖包...")
    
    # 测试ManiSkill
    try:
        import mani_skill
        print(f"✅ ManiSkill已安装，版本: {mani_skill.__version__}")
    except ImportError:
        print("❌ ManiSkill未安装，请运行: pip install mani-skill")
        return False
    
    # 测试SAPIEN
    try:
        import sapien
        print(f"✅ SAPIEN已安装，版本: {sapien.__version__}")
    except ImportError:
        print("❌ SAPIEN未安装，请运行: pip install sapien")
        return False
    
    # 测试其他依赖
    try:
        import torch
        print(f"✅ PyTorch已安装，版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import omegaconf
        print(f"✅ OmegaConf已安装，版本: {omegaconf.__version__}")
    except ImportError:
        print("❌ OmegaConf未安装")
        return False
    
    return True

def test_configuration_loading():
    """测试配置加载"""
    print("\n🔍 测试配置加载...")
    
    try:
        from omegaconf import OmegaConf
        
        # 加载仿真器配置
        simulator_config_path = project_root / "humanoidverse" / "config" / "simulator" / "maniskill_g1.yaml"
        simulator_config = OmegaConf.load(simulator_config_path)
        print("✅ 仿真器配置加载成功")
        
        # 加载环境配置
        env_config_path = project_root / "humanoidverse" / "config" / "env" / "motion_tracking_maniskill.yaml"
        env_config = OmegaConf.load(env_config_path)
        print("✅ 环境配置加载成功")
        
        # 加载实验配置
        exp_config_path = project_root / "humanoidverse" / "config" / "exp" / "motion_tracking_maniskill.yaml"
        exp_config = OmegaConf.load(exp_config_path)
        print("✅ 实验配置加载成功")
        
        # 验证配置结构
        print("   - 仿真器目标类:", simulator_config.simulator._target_)
        print("   - 环境目标类:", env_config.env._target_)
        print("   - 实验名称:", exp_config.exp.name)
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("ManiSkill G1训练配置测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        test_config_files,
        test_python_modules,
        test_dependencies,
        test_configuration_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 执行失败: {e}")
            results.append(False)
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！ManiSkill G1训练配置已准备就绪。")
        print("\n现在可以运行 'Train ManiSkill Agent' debug配置了！")
        return True
    else:
        print(f"\n❌ 有 {total - passed} 个测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
