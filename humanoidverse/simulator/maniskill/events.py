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

import torch
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger


def randomize_body_com(robot_asset, com_bias: torch.Tensor, env_ids: torch.Tensor):
    """随机化机器人身体的重心位置
    
    Args:
        robot_asset: 机器人资产
        com_bias: 重心偏差
        env_ids: 环境ID
    """
    try:
        # 这里需要根据具体的ManiSkill API来实现
        # 暂时只是记录日志
        logger.info(f"Randomizing body COM with bias: {com_bias}")
        logger.info(f"Affected environments: {env_ids}")
    except Exception as e:
        logger.warning(f"Failed to randomize body COM: {e}")


def randomize_maniskill_robot_properties(robot_asset, mass_range: list, friction_range: list, env_ids: torch.Tensor):
    """随机化ManiSkill机器人的物理属性
    
    Args:
        robot_asset: 机器人资产
        mass_range: 质量范围 [min, max]
        friction_range: 摩擦系数范围 [min, max]
        env_ids: 环境ID
    """
    try:
        # 随机化质量
        mass_scale = torch_rand_float(mass_range[0], mass_range[1], (len(env_ids), 1), device=robot_asset.device)
        logger.info(f"Randomizing robot mass with scale: {mass_scale}")
        
        # 随机化摩擦系数
        friction_scale = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=robot_asset.device)
        logger.info(f"Randomizing robot friction with scale: {friction_scale}")
        
        # 这里需要根据具体的ManiSkill API来应用这些随机化
        # 暂时只是记录日志
        
    except Exception as e:
        logger.warning(f"Failed to randomize robot properties: {e}")


def randomize_maniskill_joint_properties(robot_asset, stiffness_range: list, damping_range: list, env_ids: torch.Tensor):
    """随机化ManiSkill机器人关节的物理属性
    
    Args:
        robot_asset: 机器人资产
        stiffness_range: 刚度范围 [min, max]
        damping_range: 阻尼范围 [min, max]
        env_ids: 环境ID
    """
    try:
        # 随机化关节刚度
        stiffness_scale = torch_rand_float(stiffness_range[0], stiffness_range[1], (len(env_ids), 1), device=robot_asset.device)
        logger.info(f"Randomizing joint stiffness with scale: {stiffness_scale}")
        
        # 随机化关节阻尼
        damping_scale = torch_rand_float(damping_range[0], damping_range[1], (len(env_ids), 1), device=robot_asset.device)
        logger.info(f"Randomizing joint damping with scale: {damping_scale}")
        
        # 这里需要根据具体的ManiSkill API来应用这些随机化
        # 暂时只是记录日志
        
    except Exception as e:
        logger.warning(f"Failed to randomize joint properties: {e}")


def randomize_maniskill_task_objects(task_objects, pose_noise: float, scale_range: list, env_ids: torch.Tensor):
    """随机化ManiSkill任务对象的属性
    
    Args:
        task_objects: 任务对象列表
        pose_noise: 位姿噪声
        scale_range: 缩放范围 [min, max]
        env_ids: 环境ID
    """
    try:
        # 随机化位姿
        pose_noise_tensor = torch.randn(len(env_ids), 6, device=task_objects[0].device) * pose_noise
        logger.info(f"Randomizing task object pose with noise: {pose_noise_tensor}")
        
        # 随机化缩放
        scale_factor = torch_rand_float(scale_range[0], scale_range[1], (len(env_ids), 1), device=task_objects[0].device)
        logger.info(f"Randomizing task object scale with factor: {scale_factor}")
        
        # 这里需要根据具体的ManiSkill API来应用这些随机化
        # 暂时只是记录日志
        
    except Exception as e:
        logger.warning(f"Failed to randomize task objects: {e}")


def torch_rand_float(lower: float, upper: float, size: tuple, device: str) -> torch.Tensor:
    """生成指定范围内的随机浮点数张量
    
    Args:
        lower: 下界
        upper: 上界
        size: 张量大小
        device: 设备
        
    Returns:
        随机浮点数张量
    """
    return (upper - lower) * torch.rand(size, device=device) + lower


# 导出函数
__all__ = [
    'randomize_body_com',
    'randomize_maniskill_robot_properties',
    'randomize_maniskill_joint_properties',
    'randomize_maniskill_task_objects',
    'torch_rand_float'
]
