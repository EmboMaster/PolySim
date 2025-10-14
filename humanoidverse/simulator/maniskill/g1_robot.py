#!/usr/bin/env python3
"""
基于ManiSkill官方文档的自定义G1机器人类
配置参考：humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml
"""

import numpy as np
import sapien
import torch
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

@register_agent()
class G1HumanoidAgent(BaseAgent):
    """
    基于ManiSkill官方文档的G1人形机器人代理类
    继承自BaseAgent，通过URDF文件加载机器人
    配置完全参考humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml
    """
    
    uid = "g1_humanoid"
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 加载配置文件
        self.config: Dict[str, Any] = self._load_config()
        
        # 从配置文件设置机器人属性
        # self.urdf_path: str = self.config['robot']['asset']['asset_path']
        self.urdf_path: str = "/home/embodied/data/zxlei/embodied/humanoid/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_with_dynamics.urdf"
        # self.mjcf_path: str = "/home/embodied/data/zxlei/embodied/humanoid/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml"   
        self.fix_root_link: bool = self.config['robot']['asset']['fix_base_link']
        self.disable_self_collisions: bool = self.config['robot']['asset']['self_collisions'] == 1
        
        super().__init__(*args, **kwargs)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        # 获取配置文件路径
        config_path = Path(__file__).parent.parent.parent / "config" / "robot" / "g1" / "g1_29dof_anneal_23dof_maniskill.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @property
    def keyframes(self) -> Dict[str, Keyframe]:
        """从配置文件动态生成关键帧"""
        init_state: Dict[str, Any] = self.config['robot']['init_state']
        
        # 从配置文件读取初始位置
        pos: List[float] = init_state['pos']
        rot: List[float] = init_state['rot']  # [x, y, z, w]
        
        # 从配置文件读取默认关节角度
        default_angles: Dict[str, float] = init_state['default_joint_angles']
        dof_names: List[str] = self.config['robot']['dof_names']
        
        # 按关节名称顺序构建关节位置数组
        qpos = np.array([default_angles[joint_name] for joint_name in dof_names])
        
        # 创建姿态（注意sapien使用wxyz四元数格式）
        pose = sapien.Pose(p=pos, q=[rot[3], rot[0], rot[1], rot[2]])  # 转换为wxyz格式
        
        return {
            "standing": Keyframe(pose=pose, qpos=qpos),
            "rest": Keyframe(pose=pose, qpos=qpos)
        }
    
    @property
    def _controller_configs(self) -> Dict[str, Any]:
        """从配置文件动态生成机器人控制器配置"""
        robot_config: Dict[str, Any] = self.config['robot']
        control_config: Dict[str, Any] = robot_config['control']
        
        # 获取关节名称和限制
        dof_names: List[str] = robot_config['dof_names']
        lower_limits: List[float] = robot_config['dof_pos_lower_limit_list']
        upper_limits: List[float] = robot_config['dof_pos_upper_limit_list']
        effort_limits: List[float] = robot_config['dof_effort_limit_list']
        
        # 获取刚度配置
        stiffness_config: Dict[str, float] = control_config['stiffness']
        damping_config: Dict[str, float] = control_config['damping']
        
        # 根据关节类型分组
        leg_joints: List[str] = robot_config['lower_dof_names'][:12]  # 前12个是腿部关节
        waist_joints: List[str] = robot_config['waist_dof_names']
        arm_joints: List[str] = robot_config['upper_dof_names']
        use_delta = False
        use_target = False

        
        def get_joint_params(joint_names: List[str], stiffness_map: Dict[str, float], damping_map: Dict[str, float]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
            """根据关节名称获取参数"""
            joint_indices = [dof_names.index(name) for name in joint_names]
            print("joint_indices: ", joint_indices)
            # import pdb; pdb.set_trace()
            lower = [lower_limits[i] for i in joint_indices]
            upper = [upper_limits[i] for i in joint_indices]
            effort = [effort_limits[i] for i in joint_indices]
            
            # 根据关节类型映射刚度
            stiffness: List[float] = []
            damping: List[float] = []
            for joint_name in joint_names:
                # joint_type = joint_name.split('_')[1]  # 提取关节类型
                # if joint_type in stiffness_map:
                #     stiffness.append(stiffness_map[joint_type])
                #     damping.append(damping_map[joint_type])
                setting_flag = False
                for stiffness_key, stiffness_value in stiffness_map.items():
                    if stiffness_key in joint_name:
                        stiffness.append(stiffness_value)
                        damping.append(damping_map[stiffness_key])
                        setting_flag = True
                        break               
                if not setting_flag:
                    print(f"Default value for {joint_name}")
                    # 默认值
                    stiffness.append(100.0)
                    damping.append(2.5)
            
            return lower, upper, stiffness, damping, effort
        
        # 腿部控制器配置
        leg_lower, leg_upper, leg_stiffness, leg_damping, leg_effort = get_joint_params(
            leg_joints, stiffness_config, damping_config
        )
        leg_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=leg_joints,
            lower=leg_lower,
            upper=leg_upper,
            stiffness=leg_stiffness,
            damping=leg_damping,
            force_limit=leg_effort,
            use_delta=use_delta,
            use_target=use_target,
        )
        
        # 腰部控制器配置
        waist_lower, waist_upper, waist_stiffness, waist_damping, waist_effort = get_joint_params(
            waist_joints, stiffness_config, damping_config
        )
        waist_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=waist_joints,
            lower=waist_lower,
            upper=waist_upper,
            stiffness=waist_stiffness,
            damping=waist_damping,
            force_limit=waist_effort,
            use_delta=use_delta,
            use_target=use_target,
        )
        
        # 手臂控制器配置
        arm_lower, arm_upper, arm_stiffness, arm_damping, arm_effort = get_joint_params(
            arm_joints, stiffness_config, damping_config
        )
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=arm_joints,
            lower=arm_lower,
            upper=arm_upper,
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_effort,
            use_delta=use_delta,
            use_target=use_target,
        )
        
        # 组合控制器配置
        controller_configs = dict(
            pd_joint_pos=dict(
                legs=leg_pd_joint_pos,
                waist=waist_pd_joint_pos,
                arms=arm_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                legs=leg_pd_joint_pos,
                waist=waist_pd_joint_pos,
                arms=arm_pd_joint_pos,
            ),
        )
        
        return controller_configs
    
    def _after_init(self) -> None:
        """机器人初始化后的设置"""
        super()._after_init()
        
        # 从配置文件获取机器人属性
        robot_config: Dict[str, Any] = self.config['robot']
        
        # 设置机器人属性
        self.num_dof: int = robot_config['actions_dim']
        self.num_bodies: int = robot_config['num_bodies']
        
        # 从配置文件获取关节和身体名称
        self.dof_names: List[str] = robot_config['dof_names']
        self.body_names: List[str] = robot_config['body_names']
        
        # 获取关节和身体ID
        self.dof_ids: List[int] = list(range(self.num_dof))
        self.body_ids: List[int] = list(range(self.num_bodies))
        
        # 从配置文件获取关键身体名称
        self.key_bodies: List[str] = robot_config.get('key_bodies', [])
        self.contact_bodies: List[str] = robot_config.get('contact_bodies', [])
        
        print(f"G1 robot initialized: {self.num_dof} DOFs, {self.num_bodies} bodies")
        print(f"Joint names: {self.dof_names}")
        print(f"Body names: {self.body_names}")
        print(f"Contact bodies: {self.contact_bodies}")
        
        # 设置碰撞组以避免与地面不必要的碰撞检测
        # 基于配置文件中的contact_bodies设置
        for link in self.robot.get_links():
            if link.name in self.contact_bodies:
                # 脚部链接设置特殊的碰撞组
                link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def reset(self, init_qpos: torch.Tensor | None = None) -> None:
        """重置机器人状态"""
        super().reset(init_qpos=init_qpos)
        self.robot.set_root_linear_velocity(torch.zeros_like(self.robot.root_linear_velocity))
        self.robot.set_root_angular_velocity(torch.zeros_like(self.robot.root_angular_velocity))

        