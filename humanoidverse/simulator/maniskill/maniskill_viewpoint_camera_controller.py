# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import numpy as np
import torch
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Tuple

# ManiSkill imports
try:
    import mani_skill
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.registration import REGISTRY
    from mani_skill.utils.visualization.misc import tile_images
    from mani_skill.utils.visualization.maniskill_vis import ManiSkillVis
except ImportError:
    mani_skill = None
    BaseEnv = None
    REGISTRY = None
    ManiSkillVis = None


class ManiSkillViewportCameraController:
    """ManiSkill仿真器的视角相机控制器
    
    这个类处理仿真器中与视口相关的相机控制。它可以用于设置视角相机来跟踪不同的原点类型：
    
    - **world**: 世界中心（静态）
    - **env**: 环境中心（静态）
    - **robot**: 场景中机器人的根部（例如跟踪场景中移动的机器人）
    
    创建时，相机会设置为跟踪配置中指定的原点类型。
    
    对于robot原点类型，相机会在每个渲染步骤更新以跟踪机器人的根部位置。
    """

    def __init__(self, env: BaseEnv, cfg: dict):
        """初始化ManiSkillViewportCameraController
        
        Args:
            env: ManiSkill环境实例
            cfg: 视角相机控制器的配置
        """
        # 存储输入
        self._env = env
        self._cfg = copy.deepcopy(cfg)
        
        # 设置默认相机位置
        self.default_cam_eye = np.array(self._cfg.get('eye', [2.0, 0.0, 2.5]))
        self.default_cam_lookat = np.array(self._cfg.get('lookat', [-0.5, 0.0, 0.5]))
        self.default_cam_fov = self._cfg.get('fov', 40.0)
        
        # 相机原点类型
        self.origin_type = self._cfg.get('origin_type', 'world')
        self.env_index = self._cfg.get('env_index', 0)
        self.robot_name = self._cfg.get('robot_name', 'robot')
        
        # 相机偏移
        self.camera_offset = np.array(self._cfg.get('camera_offset', [0.0, 0.0, 0.0]))
        
        # 设置相机原点
        if self.origin_type == "env":
            # 检查环境索引是否在范围内
            self.set_view_env_index(self.env_index)
            # 将相机原点设置为环境中心
            self.update_view_to_env()
        elif self.origin_type == "robot":
            # 检查机器人名称是否设置
            if self.robot_name is None:
                raise ValueError(f"No robot name provided for viewer with origin type: '{self.origin_type}'.")
            # 设置相机跟踪机器人
            self.update_view_to_robot()
        else:
            # 将相机原点设置为世界中心
            self.update_view_to_world()
        
        # 初始化可视化器
        self._init_viewer()
        
        # 相机状态
        self._current_eye = self.default_cam_eye.copy()
        self._current_lookat = self.default_cam_lookat.copy()
        self._current_fov = self.default_cam_fov

    def _init_viewer(self):
        """初始化可视化器"""
        try:
            if ManiSkillVis is not None:
                self.viewer = ManiSkillVis()
                self.viewer.set_camera(
                    eye=self._current_eye,
                    lookat=self._current_lookat,
                    fov=self._current_fov
                )
            else:
                self.viewer = None
                print("Warning: ManiSkillVis not available, camera control limited")
        except Exception as e:
            print(f"Warning: Failed to initialize ManiSkillVis: {e}")
            self.viewer = None

    """
    属性
    """

    @property
    def cfg(self) -> dict:
        """查看器的配置"""
        return self._cfg

    @property
    def env(self) -> BaseEnv:
        """环境实例"""
        return self._env

    """
    公共函数
    """

    def set_view_env_index(self, env_index: int):
        """设置要查看的环境索引
        
        Args:
            env_index: 环境索引
        """
        if env_index < 0:
            raise ValueError(f"Environment index {env_index} is out of bounds.")
        self.env_index = env_index

    def update_view_to_world(self):
        """将视角更新到世界中心"""
        self._current_eye = self.default_cam_eye.copy()
        self._current_lookat = self.default_cam_lookat.copy()
        self._update_camera()

    def update_view_to_env(self):
        """将视角更新到环境中心"""
        if not hasattr(self._env, 'get_env_center'):
            # 如果没有获取环境中心的方法，使用默认位置
            self._current_eye = self.default_cam_eye.copy()
            self._current_lookat = self.default_cam_lookat.copy()
        else:
            env_center = self._env.get_env_center()
            if env_center is not None:
                # 计算相对于环境中心的相机位置
                self._current_eye = env_center + self.default_cam_eye
                self._current_lookat = env_center + self.default_cam_lookat
            else:
                self._current_eye = self.default_cam_eye.copy()
                self._current_lookat = self.default_cam_lookat.copy()
        
        self._update_camera()

    def update_view_to_robot(self):
        """将视角更新到机器人根部"""
        if not hasattr(self._env, 'robot') or self._env.robot is None:
            print("Warning: Robot not available, using world view")
            self.update_view_to_world()
            return
        
        robot = self._env.robot
        
        if hasattr(robot, 'get_pose'):
            robot_pos, _ = robot.get_pose()
            if robot_pos is not None:
                # 计算相对于机器人位置的相机位置
                self._current_eye = robot_pos + self.default_cam_eye + self.camera_offset
                self._current_lookat = robot_pos + self.default_cam_lookat
            else:
                self._current_eye = self.default_cam_eye.copy()
                self._current_lookat = self.default_cam_lookat.copy()
        else:
            self._current_eye = self.default_cam_eye.copy()
            self._current_lookat = self.default_cam_lookat.copy()
        
        self._update_camera()

    def update_camera_tracking(self):
        """更新相机跟踪（在每个渲染步骤调用）"""
        if self.origin_type == "robot":
            self.update_view_to_robot()
        elif self.origin_type == "env":
            self.update_view_to_env()

    def set_camera_pose(self, eye: np.ndarray, lookat: np.ndarray, fov: Optional[float] = None):
        """设置相机姿态
        
        Args:
            eye: 相机眼睛位置
            lookat: 相机看向的点
            fov: 视野角度（可选）
        """
        self._current_eye = np.array(eye)
        self._current_lookat = np.array(lookat)
        if fov is not None:
            self._current_fov = fov
        
        self._update_camera()

    def set_camera_offset(self, offset: np.ndarray):
        """设置相机偏移
        
        Args:
            offset: 相机偏移向量
        """
        self.camera_offset = np.array(offset)
        if self.origin_type == "robot":
            self.update_view_to_robot()

    def reset_camera(self):
        """重置相机到默认位置"""
        self._current_eye = self.default_cam_eye.copy()
        self._current_lookat = self.default_cam_lookat.copy()
        self._current_fov = self.default_cam_fov
        self._update_camera()

    def _update_camera(self):
        """更新相机设置"""
        if self.viewer is not None:
            try:
                self.viewer.set_camera(
                    eye=self._current_eye,
                    lookat=self._current_lookat,
                    fov=self._current_fov
                )
            except Exception as e:
                print(f"Warning: Failed to update camera: {e}")

    def get_camera_state(self) -> dict:
        """获取当前相机状态
        
        Returns:
            包含相机状态的字典
        """
        return {
            'eye': self._current_eye.copy(),
            'lookat': self._current_lookat.copy(),
            'fov': self._current_fov,
            'origin_type': self.origin_type,
            'env_index': self.env_index,
            'robot_name': self.robot_name,
            'camera_offset': self.camera_offset.copy()
        }

    def render(self, observations: Optional[list] = None):
        """渲染当前视角
        
        Args:
            observations: 观察数据列表（可选）
        """
        if self.viewer is not None:
            try:
                if observations is not None:
                    self.viewer.render(observations)
                else:
                    # 如果没有观察数据，只更新相机
                    self._update_camera()
            except Exception as e:
                print(f"Warning: Failed to render: {e}")

    def close(self):
        """关闭相机控制器"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                print(f"Warning: Failed to close viewer: {e}")
            self.viewer = None


# 默认相机配置
DEFAULT_CAMERA_CFG = {
    'eye': [2.0, 0.0, 2.5],
    'lookat': [-0.5, 0.0, 0.5],
    'fov': 40.0,
    'origin_type': 'world',
    'env_index': 0,
    'robot_name': 'robot',
    'camera_offset': [0.0, 0.0, 0.0]
}
