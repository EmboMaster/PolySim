import torch
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from termcolor import colored
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
import joblib


class ManiSkillG1MotionTracking(LeggedRobotMotionTracking):
    """ManiSkill版本的G1运动跟踪环境"""
    
    def __init__(self, config, device):
        # 初始化ManiSkill特定配置
        self._init_maniskill_config()
        
        # 调用父类初始化
        super().__init__(config, device)
        
        # 初始化ManiSkill特定功能
        self._init_maniskill_specific()
    
    def _init_maniskill_config(self):
        """初始化ManiSkill配置"""
        self.maniskill_config = getattr(self.config, 'maniskill', {})
        self.task_config = self.maniskill_config.get('task_config', {})
        self.g1_config = self.maniskill_config.get('g1_config', {})
        
        # 设置任务参数
        self.task_name = self.task_config.get('task_name', 'HumanoidLocomotion-v1')
        self.control_mode = self.task_config.get('control_mode', 'pd_joint_pos')
        self.obs_mode = self.task_config.get('obs_mode', 'state')
        self.reward_mode = self.task_config.get('reward_mode', 'dense')
        
        # 设置G1机器人参数
        self.g1_num_dof = self.g1_config.get('num_dof', 23)
        self.g1_joint_names = self.g1_config.get('joint_names', [])
        self.g1_body_names = self.g1_config.get('body_names', [])
        self.g1_base_name = self.g1_config.get('base_name', 'pelvis')
        self.g1_foot_name = self.g1_config.get('foot_name', 'ankle_roll_link')
        self.g1_knee_name = self.g1_config.get('knee_name', 'knee_link')
        self.g1_torso_name = self.g1_config.get('torso_name', 'torso_link')
        
        logger.info(f"ManiSkill G1 Motion Tracking initialized with task: {self.task_name}")
        logger.info(f"Control mode: {self.control_mode}, Obs mode: {self.obs_mode}")
        logger.info(f"G1 DOF: {self.g1_num_dof}, Joints: {len(self.g1_joint_names)}")
    
    def _init_maniskill_specific(self):
        """初始化ManiSkill特定功能"""
        # 检查仿真器是否为ManiSkill
        if hasattr(self.simulator, 'task_name'):
            logger.info("ManiSkill simulator detected, initializing task-specific features")
            self._setup_maniskill_task()
        else:
            logger.warning("Not using ManiSkill simulator, some features may not work")
    
    def _setup_maniskill_task(self):
        """设置ManiSkill任务"""
        # 这里可以添加任务特定的初始化
        # 例如：设置任务目标、初始化任务状态等
        pass
    
    def _create_envs(self):
        """创建环境 - 重写父类方法以支持ManiSkill"""
        if hasattr(self.simulator, 'create_envs'):
            # 使用ManiSkill仿真器创建环境
            logger.info("Creating environments using ManiSkill simulator")
            
            # 准备环境创建参数
            env_config = self._prepare_env_config_for_maniskill()
            
            # 创建环境
            self.simulator.create_envs(
                self.num_envs, 
                self.env_origins, 
                self.base_init_state,
                env_config
            )
        else:
            # 回退到原有逻辑
            logger.info("Using default environment creation method")
            super()._create_envs()
    
    def _prepare_env_config_for_maniskill(self):
        """为ManiSkill准备环境配置"""
        env_config = {
            'task_name': self.task_name,
            'control_mode': self.control_mode,
            'obs_mode': self.obs_mode,
            'reward_mode': self.reward_mode,
            'robot_type': 'g1_humanoid',
            'g1_config': self.g1_config
        }
        return env_config
    
    def _get_obs(self):
        """获取观察 - 重写父类方法以支持ManiSkill"""
        if hasattr(self.simulator, 'get_observations'):
            # 从ManiSkill仿真器获取观察
            logger.debug("Getting observations from ManiSkill simulator")
            maniskill_obs = self.simulator.get_observations()
            return self._convert_maniskill_obs_to_standard(maniskill_obs)
        else:
            # 回退到原有逻辑
            logger.debug("Using default observation method")
            return super()._get_obs()
    
    def _convert_maniskill_obs_to_standard(self, maniskill_obs):
        """将ManiSkill观察转换为标准格式"""
        # 根据obs_mode进行转换
        if self.obs_mode == "state":
            return self._convert_state_obs(maniskill_obs)
        elif self.obs_mode == "rgbd":
            return self._convert_rgbd_obs(maniskill_obs)
        else:
            return self._convert_default_obs(maniskill_obs)
    
    def _convert_state_obs(self, maniskill_obs):
        """转换状态观察"""
        # 这里需要根据具体的观察格式进行转换
        # 暂时返回一个基本的观察结构
        if isinstance(maniskill_obs, dict):
            # 如果观察是字典格式，提取状态信息
            state_obs = {}
            for key, value in maniskill_obs.items():
                if isinstance(value, torch.Tensor):
                    state_obs[key] = value
                else:
                    state_obs[key] = torch.tensor(value, device=self.device)
            return state_obs
        else:
            # 如果观察是张量，直接返回
            return maniskill_obs
    
    def _convert_rgbd_obs(self, maniskill_obs):
        """转换RGB-D观察"""
        # 实现RGB-D观察的转换逻辑
        # 这里需要根据具体的观察格式进行实现
        logger.warning("RGB-D observation conversion not implemented yet")
        return self._convert_state_obs(maniskill_obs)
    
    def _convert_default_obs(self, maniskill_obs):
        """转换默认观察"""
        # 实现默认观察的转换逻辑
        return self._convert_state_obs(maniskill_obs)
    
    def step(self, actions):
        """执行动作 - 重写父类方法以支持ManiSkill"""
        if hasattr(self.simulator, 'step'):
            # 使用ManiSkill仿真器执行动作
            logger.debug("Executing actions using ManiSkill simulator")
            self.simulator.step(actions)
            
            # 获取观察、奖励、终止条件
            obs = self._get_obs()
            rewards = self._compute_rewards()
            dones = self._check_termination()
            info = self._get_info()
            
            return obs, rewards, dones, info
        else:
            # 回退到原有逻辑
            logger.debug("Using default step method")
            return super().step(actions)
    
    def _compute_rewards(self):
        """计算奖励"""
        # 这里需要实现具体的奖励计算逻辑
        # 暂时返回零奖励
        rewards = torch.zeros(self.num_envs, device=self.device)
        return rewards
    
    def _check_termination(self):
        """检查终止条件"""
        # 这里需要实现具体的终止条件检查逻辑
        # 暂时返回False（不终止）
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return dones
    
    def _get_info(self):
        """获取额外信息"""
        # 返回空的信息字典
        info = {}
        return info
    
    def reset(self):
        """重置环境"""
        if hasattr(self.simulator, 'reset_envs'):
            # 使用ManiSkill仿真器重置环境
            logger.debug("Resetting environments using ManiSkill simulator")
            self.simulator.reset_envs(torch.arange(self.num_envs, device=self.device))
        else:
            # 回退到原有逻辑
            logger.debug("Using default reset method")
            return super().reset()
        
        # 获取初始观察
        obs = self._get_obs()
        return obs
    
    def render(self, sync_frame_time=True):
        """渲染环境"""
        if hasattr(self.simulator, 'render'):
            # 使用ManiSkill仿真器渲染
            self.simulator.render(sync_frame_time)
        else:
            # 回退到原有逻辑
            super().render(sync_frame_time)
    
    def close(self):
        """关闭环境"""
        if hasattr(self.simulator, 'close'):
            # 使用ManiSkill仿真器关闭
            self.simulator.close()
        else:
            # 回退到原有逻辑
            super().close()


# 导出类
__all__ = ['ManiSkillG1MotionTracking']
