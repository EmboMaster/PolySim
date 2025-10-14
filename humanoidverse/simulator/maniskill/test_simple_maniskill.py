#!/usr/bin/env python3
"""
简单的ManiSkill仿真器测试
验证基本功能是否正常
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_maniskill_simulator():
    """测试ManiSkill仿真器的基本功能"""
    print("🔍 测试ManiSkill仿真器基本功能...")
    
    try:
        # 导入ManiSkill仿真器
        from humanoidverse.simulator.maniskill.maniskill import ManiSkill
        print("✅ ManiSkill仿真器导入成功")
        
        # 创建简单的配置
        class SimpleConfig:
            def __init__(self):
                self.simulator = SimpleSimulatorConfig()
                self.robot = SimpleRobotConfig()
                self.domain_rand = {}
        
        class SimpleSimulatorConfig:
            def __init__(self):
                self.config = SimpleSimConfig()
        
        class SimpleSimConfig:
            def __init__(self):
                self.sim = SimpleSimParams()
                self.scene = SimpleSceneConfig()
                self.task = SimpleTaskConfig()
                self.g1_config = SimpleG1Config()
        
        class SimpleSimParams:
            def __init__(self):
                self.fps = 200
                self.control_decimation = 4
                self.substeps = 1
        
        class SimpleSceneConfig:
            def __init__(self):
                self.env_spacing = 4.0
        
        class SimpleTaskConfig:
            def __init__(self):
                self.task_name = "HumanoidLocomotion-v1"
                self.control_mode = "pd_joint_pos"
                self.obs_mode = "state"
                self.reward_mode = "dense"
        
        class SimpleG1Config:
            def __init__(self):
                self.num_dof = 23
                self.joint_names = [
                    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
                ]
                self.body_names = [
                    "pelvis", 
                    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
                    "waist_yaw_link", "waist_roll_link", "torso_link",
                    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
                    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link"
                ]
            
            def get(self, key, default=None):
                """模拟配置对象的get方法"""
                return getattr(self, key, default)
        
        class SimpleRobotConfig:
            def __init__(self):
                self.asset = SimpleAssetConfig()
        
        class SimpleAssetConfig:
            def __init__(self):
                self.robot_type = "g1_humanoid"
                self.asset_path = None
        
        # 创建配置实例
        config = SimpleConfig()
        
        # 创建ManiSkill仿真器实例
        device = "cpu"  # 使用CPU进行测试
        simulator = ManiSkill(config, device)
        print("✅ ManiSkill仿真器实例创建成功")
        
        # 测试load_assets方法（无参数调用）
        print("🔍 测试load_assets方法（无参数调用）...")
        try:
            num_dof, num_bodies, dof_names, body_names = simulator.load_assets()
            print(f"✅ load_assets调用成功:")
            print(f"   - DOF数量: {num_dof}")
            print(f"   - 身体部位数量: {num_bodies}")
            print(f"   - 关节名称数量: {len(dof_names)}")
            print(f"   - 身体部位名称数量: {len(body_names)}")
        except Exception as e:
            print(f"❌ load_assets调用失败: {e}")
            return False
        
        # 测试其他基本方法
        print("🔍 测试其他基本方法...")
        try:
            # 测试set_headless
            simulator.set_headless(True)
            print("✅ set_headless调用成功")
            
            # 测试setup
            simulator.setup()
            print("✅ setup调用成功")
            
            # 测试setup_terrain
            simulator.setup_terrain("plane")
            print("✅ setup_terrain调用成功")
            
        except Exception as e:
            print(f"❌ 其他方法调用失败: {e}")
            return False
        
        print("🎉 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("ManiSkill仿真器基本功能测试")
    print("=" * 60)
    
    success = test_maniskill_simulator()
    
    if success:
        print("\n🎉 测试成功！ManiSkill仿真器基本功能正常。")
        print("现在可以尝试运行'Train ManiSkill Agent'配置了！")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
