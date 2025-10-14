# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, List, Optional

# ManiSkill imports
try:
    import mani_skill
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.registration import REGISTRY
except ImportError:
    mani_skill = None
    BaseEnv = None
    REGISTRY = None

from humanoidverse.utils.torch_utils import to_torch, torch_rand_float


class ManiSkillCfg:
    """ManiSkill仿真器的配置类"""
    
    def __init__(self):
        # 环境配置
        self.episode_length_s = 3600.0
        self.substeps = 1
        self.decimation = 4
        self.action_scale = 0.25
        
        # 观察和动作空间
        self.num_actions = 19
        self.num_observations = 913
        self.observation_space = 913
        self.action_space = 19
        self.num_self_obs = 342
        self.num_ref_obs = 552
        self.num_action_obs = 19
        self.num_states = 990
        
        # 仿真参数
        self.dt = 0.005
        
        # 任务配置
        self.task_name = "PickCube-v1"
        self.control_mode = "pd_ee_delta_pose"
        self.obs_mode = "state"
        self.reward_mode = "dense"
        
        # 机器人配置
        self.robot_type = "panda"
        self.asset_path = None
        
        # 场景配置
        self.num_envs = 2
        self.env_spacing = 4.0
        self.replicate_physics = True
        
        # 地形配置
        self.terrain_type = "plane"
        self.static_friction = 1.0
        self.dynamic_friction = 1.0
        self.restitution = 0.0
        
        # 域随机化配置
        self.randomize_friction = True
        self.friction_range = [0.5, 1.5]
        self.randomize_mass = True
        self.mass_range = [0.8, 1.2]
        self.randomize_com = True
        self.com_range = [-0.1, 0.1]
        
        # 执行器配置
        self.actuators = {
            "legs": {
                "joint_names_expr": [".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
                "effort_limit": 200.0,
                "velocity_limit": 23.0,
                "stiffness": 0,
                "damping": 0,
            },
            "feet": {
                "joint_names_expr": [".*_ankle"],
                "effort_limit": 40.0,
                "velocity_limit": 9.0,
                "stiffness": 0,
                "damping": 0,
            },
            "arms": {
                "joint_names_expr": [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
                "effort_limit": 40.0,
                "velocity_limit": 9.0,
                "stiffness": 0,
                "damping": 0,
            }
        }
        
        # 机器人身体名称
        self.body_names = [
            "base_link",
            "link1",
            "link2", 
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
        ]
        
        # 关节名称
        self.joint_names = [
            "joint1",
            "joint2", 
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        
        # 基座名称
        self.base_name = "base_link"
        
        # 手部名称
        self.hand_name = "hand"
        
        # 扩展链接配置
        self.extend_body_parent_ids = [0, 1, 2]
        self.extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])
        
        # 遥操作关键点
        self.teleop_selected_keypoints_names = [
            "base_link",
            "link1",
            "link2",
            "link3", 
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
        ]


class ManiSkillArticulationCfg:
    """ManiSkill机器人关节配置"""
    
    def __init__(self):
        # 机器人资产配置
        self.usd_path = None  # 可以设置为自定义USD文件路径
        self.activate_contact_sensors = True
        
        # 刚体属性
        self.disable_gravity = False
        self.retain_accelerations = False
        self.linear_damping = 0.0
        self.angular_damping = 0.0
        self.max_linear_velocity = 1000.0
        self.max_angular_velocity = 1000.0
        self.max_depenetration_velocity = 1.0
        
        # 关节属性
        self.enabled_self_collisions = False
        self.solver_position_iteration_count = 4
        self.solver_velocity_iteration_count = 4
        
        # 初始状态
        self.init_pos = (0.0, 0.0, 1.05)
        self.init_joint_pos = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }
        self.init_joint_vel = {".*": 0.0}
        
        # 软关节限制因子
        self.soft_joint_pos_limit_factor = 0.9
        
        # 执行器配置
        self.actuators = {
            "main": {
                "joint_names_expr": [".*"],
                "effort_limit": 100.0,
                "velocity_limit": 50.0,
                "stiffness": {
                    "joint1": 100.0,
                    "joint2": 100.0,
                    "joint3": 100.0,
                    "joint4": 100.0,
                    "joint5": 100.0,
                    "joint6": 100.0,
                    "joint7": 100.0,
                },
                "damping": {
                    "joint1": 10.0,
                    "joint2": 10.0,
                    "joint3": 10.0,
                    "joint4": 10.0,
                    "joint5": 10.0,
                    "joint6": 10.0,
                    "joint7": 10.0,
                },
            }
        }


# 默认配置实例
MANISKILL_CFG = ManiSkillCfg()
MANISKILL_ARTICULATION_CFG = ManiSkillArticulationCfg()
