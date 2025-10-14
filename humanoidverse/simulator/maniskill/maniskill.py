import sys
import os
from loguru import logger
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import traceback
from mani_skill.utils.structs import SimConfig
import mani_skill

from mani_skill.envs.sapien_env import BaseEnv
# 根据ManiSkill文档，我们需要继承BaseAgent来创建自定义机器人
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils.registration import register_env
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Articulation

import sapien
MANISKILL_AVAILABLE = True
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
from humanoidverse.simulator.maniskill.maniskill_cfg import MANISKILL_CFG, MANISKILL_ARTICULATION_CFG
from humanoidverse.simulator.maniskill.event_cfg import DEFAULT_MANISKILL_EVENT_CFG
from humanoidverse.simulator.maniskill.events import (
    randomize_maniskill_robot_properties,
    randomize_maniskill_joint_properties,
    randomize_maniskill_task_objects
)
from humanoidverse.simulator.maniskill.maniskill_viewpoint_camera_controller import (
    ManiSkillViewportCameraController,
    DEFAULT_CAMERA_CFG
)
from humanoidverse.simulator.maniskill.g1_robot import G1HumanoidAgent

class ManiSkill(BaseSimulator):
    """
    ManiSkill仿真器实现，基于ManiSkill3框架提供机器人操作仿真环境
    参考IsaacSim的实现逻辑：场景创建 -> 机器人加载 -> 资产管理 -> 关节查找
    """
    
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        
        # 加载默认配置
        self.default_cfg = MANISKILL_CFG
        self.default_articulation_cfg = MANISKILL_ARTICULATION_CFG
        
        # ManiSkill specific attributes - 参考IsaacSim的结构
        self.scene = None
        
        # Simulation parameters - 参考IsaacSim
        self.sim_dt = 1.0 / self.simulator_config.sim.fps
        self.control_decimation = self.simulator_config.sim.control_decimation
        self.substeps = self.simulator_config.sim.substeps
        
        # Device configuration
        self.device = device
        self.headless = False
        
        # 从配置中读取headless设置
        if hasattr(config, 'headless'):
            self.headless = config.headless
            logger.info(f"ManiSkill simulator headless mode: {self.headless}")
        
        # 确保设备一致性 - 如果指定了CUDA设备，但实际在CPU上运行，则使用CPU
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"CUDA device {self.device} requested but not available. Using CPU instead.")
            self.device = 'cpu'
        elif self.device.startswith('cuda'):
            # 确保CUDA设备可用
            try:
                torch.cuda.set_device(self.device)
                logger.info(f"Using CUDA device: {self.device}")
            except Exception as e:
                logger.warning(f"Failed to set CUDA device {self.device}: {e}. Using CPU instead.")
                self.device = 'cpu'
        
        # State buffers - 参考IsaacSim
        # 这些变量将只包含配置文件中body_names指定的有效身体部位
        self._rigid_body_pos: Optional[torch.Tensor] = None
        self._rigid_body_rot: Optional[torch.Tensor] = None
        self._rigid_body_vel: Optional[torch.Tensor] = None
        self._rigid_body_ang_vel: Optional[torch.Tensor] = None
        
        # DOF and body information - 参考IsaacSim
        self.num_dof = 0
        self.num_bodies = 0
        self.dof_names = []
        self.body_names = []
        self.dof_ids = []
        self.body_ids = []
        
        # 有效身体部位名称 - 从配置文件中读取
        self._valid_body_names = []
        
        # 身体名称到索引的映射
        self._body_name_to_idx = {}
        
        # Environment origins and spacing - 参考IsaacSim
        self.env_origins = None
        self.env_spacing = getattr(self.simulator_config.scene, 'env_spacing', 5.0)
        self.num_envs = getattr(self.simulator_config.scene, 'num_envs', 1)
        
        # 事件管理器
        self.events_cfg = DEFAULT_MANISKILL_EVENT_CFG
        self.event_manager = None
        
        # 相机控制器
        self.viewport_camera_controller = None
        
        # Viewer attributes - 参考Genesis和IsaacGym的实现
        self.viewer = None
        self.visualize_viewer = False
        self.enable_viewer_sync = True
        
        # 任务配置
        self.task_name = getattr(self.simulator_config.task, 'task_name', 'HumanoidLocomotion-v1')
        self.control_mode = getattr(self.simulator_config.task, 'control_mode', 'pd_joint_pos')
        self.obs_mode = getattr(self.simulator_config.task, 'obs_mode', 'state')
        self.reward_mode = getattr(self.simulator_config.task, 'reward_mode', 'dense')
        
        # 域随机化配置
        self.domain_rand_config = getattr(config, 'domain_rand', {})
        
        # 默认重心和偏差 - 参考IsaacSim
        self.default_coms = None
        self.base_com_bias = None
        
        # 添加训练脚本期望的属性
        self.base_quat = None
        self.base_pos = None
        self.base_lin_vel = None
        self.base_ang_vel = None
        self.dof_pos = None
        self.dof_vel = None
        
        # 添加缺失的属性以兼容训练脚本
        self.all_root_states = None
        self.robot_root_states = None
        self.contact_forces = None
        self.terrain = None
        
        # 初始化base_init_state属性
        self.base_init_state = None
        
        # 添加_body_list属性以兼容训练脚本
        self._body_list = []
        
        # 如果配置中包含G1配置，则立即加载资产
        if hasattr(self, 'simulator_config') and hasattr(self.simulator_config, 'g1_config'):
            try:
                self._load_g1_humanoid_assets(None)
                logger.info("G1 humanoid assets loaded during initialization")
            except Exception as e:
                logger.warning(f"Failed to load G1 assets during initialization: {e}")
        
        logger.info("ManiSkill simulator initialized")
        # self.create_envs()
        # _obs, _info = self._env.reset(seed=[2022 + i for i in range(self.num_envs)], options=dict(reconfigure=True))
        # self._env.step(action=None)

    def set_headless(self, headless):
        """设置无头模式"""
        super().set_headless(headless)
        self.headless = headless

    def setup(self):
        """设置仿真器 - 参考IsaacSim的setup逻辑"""
        logger.info("Setting up ManiSkill simulator...")
        
        # 设置事件管理器
        self._setup_event_manager()
        
        # 设置相机控制器
        self._setup_camera_controller()
        
        logger.info("ManiSkill simulator setup completed")

    def _setup_event_manager(self):
        """设置事件管理器 - 参考IsaacSim的事件管理逻辑"""
        try:
            # 配置域随机化事件
            if self.domain_rand_config.get("randomize_link_mass", False):
                logger.info("Setting up link mass randomization")
            
            if self.domain_rand_config.get("randomize_friction", False):
                logger.info("Setting up friction randomization")
            
            if self.domain_rand_config.get("randomize_base_com", False):
                logger.info("Setting up base COM randomization")
            
            # 创建简单的事件管理器（不依赖omni）
            self.event_manager = self._create_simple_event_manager()
            logger.info("Simple event manager setup completed")
                
        except Exception as e:
            logger.warning(f"Event manager setup failed: {e}")
            self.event_manager = None

    def _create_simple_event_manager(self):
        """创建简单的事件管理器"""
        class SimpleEventManager:
            def __init__(self, events_cfg, simulator):
                self.events_cfg = events_cfg
                self.simulator = simulator
                self.available_modes = ["startup"]
            
            def apply(self, mode="startup"):
                if mode == "startup":
                    logger.info("Applying startup events")
                    # 这里可以添加启动时的事件处理逻辑
                    pass
        
        return SimpleEventManager(self.events_cfg, self)

    def _setup_camera_controller(self):
        """设置相机控制器"""
        if not self.headless:
            try:
                # 创建相机控制器
                self.viewport_camera_controller = ManiSkillViewportCameraController(
                    self, DEFAULT_CAMERA_CFG
                )
                logger.info("Camera controller setup completed")
            except Exception as e:
                logger.warning(f"Failed to setup camera controller: {e}")
                self.viewport_camera_controller = None

    def setup_terrain(self, mesh_type):
        """配置地形 - 参考IsaacSim的地形设置逻辑"""
        if mesh_type == 'plane':
            logger.info("Setting up plane terrain")
            # ManiSkill中地形通常在环境创建时设置
        elif mesh_type == 'heightfield':
            logger.info("Setting up heightfield terrain")
            # 支持高度场地形
        elif mesh_type == 'trimesh':
            logger.info("Setting up trimesh terrain")
            # 支持三角网格地形
        else:
            logger.warning(f"Terrain type {mesh_type} not supported in ManiSkill")

    def load_assets(self, robot_config=None):
        """加载机器人资产 - 参考IsaacSim的load_assets逻辑"""
        logger.info("Loading robot assets...")
        
        # 如果没有提供robot_config，尝试从环境配置中获取
        if robot_config is None:
            if hasattr(self, 'robot_config') and self.robot_config is not None:
                robot_config = self.robot_config
            else:
                logger.warning("No robot_config provided, using default G1 configuration")
                # 使用默认的G1配置
                return self._load_g1_humanoid_assets(None)
        
        # 从配置中获取机器人信息 - 参考IsaacSim的配置读取方式
        robot_type = robot_config.asset.robot_type
        robot_path = robot_config.asset.asset_path
        
        # 特殊处理G1机器人
        if robot_type == "g1_humanoid" or "g1" in robot_type:
            return self._load_g1_humanoid_assets(robot_config)
        
        # 加载机器人URDF或ManiSkill资产
        if robot_path and os.path.exists(robot_path):
            logger.info(f"Loading robot from: {robot_path}")
            # 这里需要根据ManiSkill的API加载机器人
            # self.robot_assets[robot_type] = load_robot_asset(robot_path)
        else:
            logger.warning(f"Robot asset not found at: {robot_path}")
        
        # 获取机器人属性
        # self.num_dof = get_robot_dof_count(self.robot_assets[robot_type])
        # self.num_bodies = get_robot_body_count(self.robot_assets[robot_type])
        # self.dof_names = get_robot_dof_names(self.robot_assets[robot_type])
        # self.body_names = get_robot_body_names(self.robot_assets[robot_type])
        
        logger.info(f"Robot loaded: {self.num_dof} DOFs, {self.num_bodies} bodies")
        return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def _load_g1_humanoid_assets(self, robot_config):
        """加载G1人形机器人资产 - 参考IsaacSim的资产加载逻辑"""
        logger.info("Loading G1 humanoid robot assets...")
        
        # 从仿真器配置中获取G1配置
        g1_config = getattr(self.simulator_config, 'g1_config', {})
        
        # 设置G1机器人的属性
        # 使用配置文件中的DOF数量，而不是ManiSkill3的实际DOF数量
        # 这样可以保持与现有配置文件的兼容性
        self.num_dof = g1_config.get('num_dof', 23)  # 使用配置文件中的DOF数量
        
        # 从配置文件中读取有效的身体部位名称
        # 这些是训练脚本期望的有效身体部位，不是所有关节
        self._valid_body_names = g1_config.get('body_names', [
            "pelvis", 
            "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
            "waist_yaw_link", "waist_roll_link", "torso_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link"
        ])
        
        # 设置num_bodies为有效身体部位的数量
        self.num_bodies = len(self._valid_body_names)
        
        # 创建身体名称到索引的映射
        self._body_name_to_idx = {name: idx for idx, name in enumerate(self._valid_body_names)}
        
        # G1关节名称 - 这些是实际的关节，用于控制
        self.dof_names = g1_config.get('joint_names', [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ])
        
        # 设置body_names为有效身体部位名称（与训练脚本期望的一致）
        self.body_names = self._valid_body_names.copy()
        
        # 设置_body_list属性以兼容训练脚本
        self._body_list = self.body_names.copy()
        
        logger.info(f"G1 robot loaded: {self.num_dof} DOFs, {self.num_bodies} valid bodies")
        logger.info(f"Valid body names: {self._valid_body_names}")
        logger.info(f"Joint names: {self.dof_names}")
        
        return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def find_rigid_body_indice(self, body_name):
        """查找刚体索引 - 参考IsaacSim的find_rigid_body_indice逻辑"""
        try:
            # 使用有效身体部位的索引映射
            if body_name in self._body_name_to_idx:
                return self._body_name_to_idx[body_name]
            else:
                logger.warning(f"Body name {body_name} not found in valid body names: {self._valid_body_names}")
                return -1
        except Exception as e:
            logger.error(f"Failed to find rigid body index for {body_name}: {e}")
            return -1

    def _create_g1_robot(self):
        """创建G1机器人 - 使用ManiSkill官方文档推荐的自定义机器人类"""
        try:
            # 导入自定义的G1机器人类
            from humanoidverse.simulator.maniskill.g1_robot import G1HumanoidAgent
            
            # 创建G1机器人代理
            self._env.agent = G1HumanoidAgent(
                scene=self.scene,
                control_freq=self.simulator_config.sim.fps // self.simulator_config.sim.control_decimation,
                control_mode=self.control_mode,
                initial_pose=None  # 将在_set_robot_initial_state中设置
            )
            
            # 获取机器人对象
            self._env.agent.robot = self._env.agent.robot
            
            # 从机器人代理获取DOF相关属性
            self.num_dof = self._env.agent.num_dof
            self.dof_names = self._env.agent.dof_names
            self.dof_ids = self._env.agent.dof_ids
            
            # 从配置文件中读取有效的身体部位名称
            # 这些是训练脚本期望的有效身体部位，不是所有关节
            g1_config = getattr(self.simulator_config, 'g1_config', {})
            self._valid_body_names = g1_config.get('body_names', [
                "pelvis", 
                "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
                "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
                "waist_yaw_link", "waist_roll_link", "torso_link",
                "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
                "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link"
            ])
            
            # 设置num_bodies为有效身体部位的数量
            self.num_bodies = len(self._valid_body_names)
            
            # 创建身体名称到索引的映射
            self._body_name_to_idx = {name: idx for idx, name in enumerate(self._valid_body_names)}
            
            # 设置body_names为有效身体部位名称（与训练脚本期望的一致）
            self.body_names = self._valid_body_names.copy()
            
            # 设置body_ids为有效身体部位的索引
            self.body_ids = list(range(self.num_bodies))
            
            logger.info("G1 robot created successfully using custom agent class")
            
        except Exception as e:
            logger.error(f"Failed to create G1 robot: {e}")
            raise
    
    def _create_robot_manually(self, urdf_path, sub_scene):
        """手动创建机器人（当URDF加载器失败时的备选方案）"""
        try:
            import xml.etree.ElementTree as ET
            
            # 解析URDF文件
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # 创建机器人构建器
            robot_builder = sub_scene.create_articulation_builder()
            
            # 解析链接和关节
            links = {}
            joints = []
            
            # 首先解析所有链接
            for link_elem in root.findall('link'):
                link_name = link_elem.get('name')
                links[link_name] = link_elem
            
            # 然后解析所有关节
            for joint_elem in root.findall('joint'):
                joint_name = joint_elem.get('name')
                joint_type = joint_elem.get('type')
                parent_link = joint_elem.find('parent').get('link')
                child_link = joint_elem.find('child').get('link')
                
                joints.append({
                    'name': joint_name,
                    'type': joint_type,
                    'parent': parent_link,
                    'child': child_link,
                    'elem': joint_elem
                })
            
            # 创建机器人结构
            # 这里需要根据URDF的具体结构来创建
            # 由于G1机器人的URDF结构比较复杂，这里提供一个简化的实现
            
            logger.warning("Manual URDF parsing not fully implemented for complex robots like G1")
            logger.info("Creating a simplified robot structure...")
            
            # 创建一个简化的机器人结构
            # 这里需要根据实际的URDF内容来实现
            # 暂时创建一个基本的关节结构
            
            # 设置基本的机器人属性
            self.num_dof = 23  # G1的默认关节数
            self.num_bodies = 24  # G1的默认身体数
            self.dof_names = [f"joint_{i}" for i in range(self.num_dof)]
            self.body_names = [f"link_{i}" for i in range(self.num_bodies)]
            self.dof_ids = list(range(self.num_dof))
            self.body_ids = list(range(self.num_bodies))
            
            # 创建一个占位符机器人对象
            # 这里需要根据实际的URDF结构来创建真正的机器人
            self._env.agent.robot = None
            
            logger.warning("Manual robot creation not fully implemented. Robot object is None.")
            
        except Exception as e:
            logger.error(f"Failed to create robot manually: {e}")
            raise

    def _set_robot_initial_state(self):
        """设置机器人初始状态 - 参考IsaacSim的状态设置逻辑"""
        if self._env.agent.robot is None or self.base_init_state is None:
            return
        
        # 设置基座位置和旋转
        base_pos = self.base_init_state[:3]      # 基座位置 (x, y, z)
        base_rot = self.base_init_state[3:7][:, [1, 2, 3, 0]]     # 基座旋转四元数 (qx, qy, qz, qw)
        
        # 设置机器人位姿
        # if hasattr(self._env.agent.robot, 'set_pose'):
            # 转换torch张量为numpy数组
        p_tensor = base_pos.detach().cpu() if hasattr(base_pos, 'detach') else torch.tensor(base_pos, dtype=torch.float)
        q_tensor = base_rot.detach().cpu() if hasattr(base_rot, 'detach') else torch.tensor(base_rot, dtype=torch.float)
        self._env.agent.robot.set_pose(sapien.Pose(p=p_tensor, q=q_tensor))
        logger.info(f"Set robot base pose: pos={p_tensor}, rot={q_tensor}")
        
        # 设置关节位置为全0
        # if hasattr(self._env.agent.robot, 'set_qpos'):

        # 创建全0的关节位置数组
        zero_joint_pos = torch.zeros(self.num_dof, dtype=torch.float)
        self._env.agent.robot.set_qpos(zero_joint_pos)
        logger.info(f"Set joint positions to zero for {self.num_dof} DOFs")

        
        logger.info("Robot initial state set successfully")


    def get_dof_limits_properties(self):
        """
        获取DOF限制和属性 - 参考Genesis模拟器的实现
        创建并返回位置限制、速度限制和扭矩限制的张量
        
        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        if self._env.agent.robot is None:
            logger.warning("No robot created yet")
            return None, None, None
        
        # 初始化限制张量 - 参考Genesis的实现
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 从配置文件获取所有限制信息
        # 参考 Genesis 的实现：从 robot_cfg 中读取
        for i in range(self.num_dof):
            # 设置位置限制 - 从配置文件读取
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]  # 下界
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]  # 上界
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]       # 下界
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]       # 上界
            
            # 从配置文件读取速度限制和扭矩限制
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
        
        # 应用软限制 - 参考Genesis的实现
        soft_dof_pos_limit = self.env_config.rewards.reward_limit.soft_dof_pos_limit
        
        for i in range(self.num_dof):
            # 计算软限制
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 中点
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]        # 范围
            
            # 应用软限制系数
            self.dof_pos_limits[i, 0] = m - 0.5 * r * soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * soft_dof_pos_limit
        
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits









    def find_rigid_body_indice(self, body_name):
        """查找刚体索引 - 参考IsaacSim的find_rigid_body_indice逻辑"""
        try:
            # 使用有效身体部位的索引映射
            if body_name in self._body_name_to_idx:
                return self._body_name_to_idx[body_name]
            else:
                logger.warning(f"Body name {body_name} not found in valid body names: {self._valid_body_names}")
                return -1
        except Exception as e:
            logger.error(f"Failed to find rigid body index for {body_name}: {e}")
            return -1

    def prepare_sim(self):
        """准备仿真 - 参考IsaacSim的仿真准备逻辑"""
        logger.info("Preparing ManiSkill simulation...")
    
        # 初始化场景 - ManiSkill场景不需要显式初始化
        if self.scene is not None:
            logger.info("Scene ready")
        
        # 应用启动事件
        if self.event_manager is not None:
            self.event_manager.apply(mode="startup")
        
        # 初始化张量 - 参考IsaacSim的逻辑
        self.refresh_sim_tensors()
        
        logger.info("Simulation preparation completed")

    def refresh_sim_tensors(self):
        """刷新仿真张量 - 直接读取机器人状态"""
        # 1. 获取机器人基座状态
        robot_pose = self._env.agent.robot.pose
        pose_p = robot_pose.p
        pose_q = robot_pose.q
        
        # 转换为torch张量并扩展到多环境
        self.base_pos = torch.tensor(pose_p, device=self.device)
        self.base_quat = torch.tensor(pose_q[:, [1, 2, 3, 0]], device=self.device)
        
        # 2. 获取关节状态
        self.dof_pos = torch.tensor(self._env.agent.robot.qpos, device=self.device)
        self.dof_vel = torch.tensor(self._env.agent.robot.qvel, device=self.device)
        
        # 3. 更新刚体状态
        if not hasattr(self, '_rigid_body_pos') or self._rigid_body_pos is None:
            self._init_state_buffers()
        self._update_rigid_body_states()
        
        # 4. 获取接触力
        contact_forces = self._env.agent.robot.get_net_contact_forces(self._valid_body_names)
        self.contact_forces = torch.tensor(contact_forces, device=self.device)
        if len(self.contact_forces.shape) == 2:
            self.contact_forces = self.contact_forces.unsqueeze(0).expand(self.num_envs, -1, -1)
        
        # 5. 初始化根状态张量（如果需要）
        if not hasattr(self, 'all_root_states') or self.all_root_states is None:
            self.all_root_states = torch.zeros(self.num_envs, 13, device=self.device)
        if not hasattr(self, 'robot_root_states') or self.robot_root_states is None:
            self.robot_root_states = torch.zeros(self.num_envs, 13, device=self.device)
        
        # 6. 更新根状态张量：位置(3) + 旋转(4) + 线速度(3) + 角速度(3)
        self.robot_root_states[:, :3] = self.base_pos
        self.robot_root_states[:, 3:7] = self.base_quat
        
        # 获取基座速度
        root_lin_vel = self._env.agent.robot.root_linear_velocity
        root_ang_vel = self._env.agent.robot.root_angular_velocity
        
        lin_vel = torch.tensor(root_lin_vel, device=self.device)
        ang_vel = torch.tensor(root_ang_vel, device=self.device)
        
        self.robot_root_states[:, 7:10] = lin_vel
        self.robot_root_states[:, 10:13] = ang_vel
        
        # 更新基座速度
        self.base_lin_vel = lin_vel
        self.base_ang_vel = ang_vel
        
        # all_root_states 目前只包含机器人
        self.all_root_states = self.robot_root_states.clone()
        


    def _update_state_buffers(self):
        """更新状态缓冲区 - 参考IsaacSim的状态缓冲区更新逻辑"""
        if self._env.agent.robot is None:
            logger.warning("No robot available, cannot update state buffers")
            return
        
        # 获取刚体状态
        if not hasattr(self._env.agent.robot, 'get_links'):
            logger.warning("Robot does not have 'get_links' method")
            return
        
        links = self._env.agent.robot.get_links()
        if not links:
            logger.warning("Robot links list is empty")
            return
        
        # 创建链接名称到链接对象的映射
        link_name_to_link = {link.name: link for link in links}
        
        # 只更新配置文件中body_names指定的有效身体部位状态
        positions = []
        rotations = []
        velocities = []
        angular_velocities = []
        
        for body_name in self._valid_body_names:
            if body_name in link_name_to_link:
                link = link_name_to_link[body_name]
                
                if hasattr(link, 'pose') and link.pose is not None:
                    positions.append(link.pose.p)
                    rotations.append(link.pose.q)
                else:
                    logger.warning(f"Link {body_name} pose is None")
                    positions.append([0.0, 0.0, 0.0])
                    rotations.append([0.0, 0.0, 0.0, 1.0])
                
                if hasattr(link, 'get_velocity'):
                    vel = link.get_velocity()
                    if vel is not None and len(vel) >= 6:
                        velocities.append(vel[:3])
                        angular_velocities.append(vel[3:6])
                    else:
                        logger.warning(f"Link {body_name} velocity is None or invalid")
                        velocities.append([0.0, 0.0, 0.0])
                        angular_velocities.append([0.0, 0.0, 0.0])
                else:
                    velocities.append([0.0, 0.0, 0.0])
                    angular_velocities.append([0.0, 0.0, 0.0])
            else:
                logger.warning(f"Body {body_name} not found in robot links. Available links: {list(link_name_to_link.keys())}")
                # 如果找不到对应的链接，使用零值填充
                positions.append([0.0, 0.0, 0.0])
                rotations.append([0.0, 0.0, 0.0, 1.0])
                velocities.append([0.0, 0.0, 0.0])
                angular_velocities.append([0.0, 0.0, 0.0])
        
        # 转换为torch张量并扩展到多环境
        if positions:
            positions_tensor = torch.tensor(positions, device=self.device)
            if len(positions_tensor.shape) == 2:
                positions_tensor = positions_tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
            self._rigid_body_pos = positions_tensor
        
        if rotations:
            rotations_tensor = torch.tensor(rotations, device=self.device)
            if len(rotations_tensor.shape) == 2:
                rotations_tensor = rotations_tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
            self._rigid_body_rot = rotations_tensor[:, [1, 2, 3, 0]]
        
        if velocities:
            velocities_tensor = torch.tensor(velocities, device=self.device)
            if len(velocities_tensor.shape) == 2:
                velocities_tensor = velocities_tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
            self._rigid_body_vel = velocities_tensor
        
        if angular_velocities:
            angular_velocities_tensor = torch.tensor(angular_velocities, device=self.device)
            if len(angular_velocities_tensor.shape) == 2:
                angular_velocities_tensor = angular_velocities_tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
            self._rigid_body_ang_vel = angular_velocities_tensor

    def _init_state_buffers(self):
        """初始化状态缓冲区 - 参考IsaacSim的缓冲区初始化逻辑"""
        if self.num_bodies == 0:
            return
            
        # 初始化状态缓冲区 - 只包含配置文件中body_names指定的有效身体部位
        self._rigid_body_pos = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._rigid_body_rot = torch.zeros(self.num_envs, self.num_bodies, 4, device=self.device)
        self._rigid_body_vel = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._rigid_body_ang_vel = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        
        logger.info(f"Initialized state buffers for {self.num_bodies} valid bodies: {self._valid_body_names}")

    def _init_basic_tensors(self):
        """初始化基本张量 - 即使刷新失败也要确保基本张量存在"""
        try:
            # 初始化基本张量 - 确保所有张量都在正确的设备上
            if not hasattr(self, 'dof_pos') or self.dof_pos is None:
                self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            
            if not hasattr(self, 'dof_vel') or self.dof_vel is None:
                self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            
            if not hasattr(self, 'base_pos') or self.base_pos is None:
                self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
            
            if not hasattr(self, 'base_lin_vel') or self.base_lin_vel is None:
                self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
            
            if not hasattr(self, 'base_ang_vel') or self.base_ang_vel is None:
                self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
            
            if not hasattr(self, 'base_quat') or self.base_quat is None:
                self.base_quat = torch.tensor([1, 0., 0., 0], device=self.device)
            
            if not hasattr(self, 'robot_root_states') or self.robot_root_states is None:
                self.robot_root_states = torch.zeros(self.num_envs, 13, device=self.device)
            
            # 确保所有张量都在正确的设备上
            self._ensure_tensors_on_device()
            
            logger.info("Basic tensors initialized on device: " + str(self.device))
            
        except Exception as e:
            logger.error(f"Failed to initialize basic tensors: {e}")

    def _ensure_tensors_on_device(self):
        """确保所有张量都在正确的设备上"""
        try:
            # 检查并移动所有张量到正确的设备
            tensor_attrs = [
                'dof_pos', 'dof_vel', 'base_pos', 'base_lin_vel', 'base_ang_vel', 
                'base_quat', 'robot_root_states', '_rigid_body_pos', '_rigid_body_rot',
                '_rigid_body_vel', '_rigid_body_ang_vel', 'contact_forces', 'all_root_states'
            ]
            
            for attr in tensor_attrs:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    tensor = getattr(self, attr)
                    if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                        logger.info(f"Moving {attr} from {tensor.device} to {self.device}")
                        setattr(self, attr, tensor.to(self.device))
            
            logger.info("All tensors moved to device: " + str(self.device))
            
        except Exception as e:
            logger.error(f"Failed to ensure tensors on device: {e}")

    def _update_rigid_body_states(self):
        """更新刚体状态 - 参考IsaacSim的刚体状态更新逻辑"""
        # 确保所有刚体状态都被更新，即使某些部分失败也要继续
        
        # 如果没有机器人，直接返回，让错误暴露出来
        if self._env.agent.robot is None:
            logger.error("No robot available, cannot update rigid body states")
            return
        
        # 获取机器人链接
        links = getattr(self._env.agent.robot, 'get_links', lambda: [])()
        
        # 创建链接名称到链接对象的映射
        link_name_to_link = {link.name: link for link in links}
        
        # 只更新配置文件中body_names指定的有效身体部位状态
        for body_idx, body_name in enumerate(self._valid_body_names):
            if body_name in link_name_to_link:
                link = link_name_to_link[body_name]
                
                # 更新位置和旋转
                link_pose = getattr(link, 'pose', None)
                if link_pose is None:
                    logger.error(f"Link {body_name} has no pose attribute")
                    continue
                
                pose_p = getattr(link_pose, 'p', None)
                pose_q = getattr(link_pose, 'q', None)
                
                if pose_p is None or pose_q is None:
                    logger.error(f"Link {body_name} pose position or rotation is None")
                    continue
                
                # 位置
                pose_p_tensor = torch.tensor(pose_p, device=self.device)
                if len(pose_p_tensor.shape) == 1:
                    pose_p_tensor = pose_p_tensor.unsqueeze(0).expand(self.num_envs, -1)
                self._rigid_body_pos[:, body_idx, :] = pose_p_tensor
                
                # 旋转（保持xyzw格式）
                pose_q_tensor = torch.tensor(pose_q, device=self.device)
                if len(pose_q_tensor.shape) == 1:
                    pose_q_tensor = pose_q_tensor.unsqueeze(0).expand(self.num_envs, -1)
                self._rigid_body_rot[:, body_idx, :] = pose_q_tensor[:, [1, 2, 3, 0]]
                
                # 更新速度
                # 获取线速度
                get_linear_velocity_func = getattr(link, 'get_linear_velocity', None)
                if get_linear_velocity_func is None:
                    logger.error(f"Link {body_name} has no get_linear_velocity method")
                    continue
                
                lin_vel = get_linear_velocity_func()
                if lin_vel is None:
                    logger.error(f"Link {body_name} linear velocity is None")
                    continue
                
                # 获取角速度
                get_angular_velocity_func = getattr(link, 'get_angular_velocity', None)
                if get_angular_velocity_func is None:
                    logger.error(f"Link {body_name} has no get_angular_velocity method")
                    continue
                
                ang_vel = get_angular_velocity_func()
                if ang_vel is None:
                    logger.error(f"Link {body_name} angular velocity is None")
                    continue
                
                # 线速度
                lin_vel_tensor = torch.tensor(lin_vel, device=self.device)
                if len(lin_vel_tensor.shape) == 1:
                    lin_vel_tensor = lin_vel_tensor.unsqueeze(0).expand(self.num_envs, -1)
                self._rigid_body_vel[:, body_idx, :] = lin_vel_tensor
                
                # 角速度
                ang_vel_tensor = torch.tensor(ang_vel, device=self.device)
                if len(ang_vel_tensor.shape) == 1:
                    ang_vel_tensor = ang_vel_tensor.unsqueeze(0).expand(self.num_envs, -1)
                self._rigid_body_ang_vel[:, body_idx, :] = ang_vel_tensor
            else:
                logger.error(f"Body {body_name} not found in robot links. Available links: {list(link_name_to_link.keys())}")
                # 不填充零值，让错误暴露出来
        
        logger.debug(f"Updated rigid body states for {len(self._valid_body_names)} valid bodies")

    def apply_torques_at_dof(self, torques):
        """在DOF上应用扭矩 - 参考IsaacSim的扭矩应用逻辑"""
        # self._env.agent.robot.set_qf(torques)
        # self._env.scene._gpu_apply_all()
        # self._env.scene._gpu_fetch_all()
        pass
        
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """设置Actor根状态张量 - 支持多环境并行"""
        root_pose_modified = self._env.agent.robot.get_root_pose()
        root_linear_velocity_modified = self._env.agent.robot.get_root_linear_velocity()
        root_angular_velocity_modified = self._env.agent.robot.get_root_angular_velocity()
        root_pose_modified.p[set_env_ids] = root_states[set_env_ids, :3]
        root_pose_modified.q[set_env_ids] = root_states[set_env_ids, 3:7]
        root_linear_velocity_modified[set_env_ids] = root_states[set_env_ids, 7:10]
        root_angular_velocity_modified[set_env_ids] = root_states[set_env_ids, 10:13]
        self._env.agent.robot.set_root_pose(root_pose_modified)
        self._env.agent.robot.set_root_linear_velocity(root_linear_velocity_modified)
        self._env.agent.robot.set_root_angular_velocity(root_angular_velocity_modified)
        self._env.scene._gpu_apply_all()
        self._env.scene._gpu_fetch_all()
            # 批量设置所有环境的状态


    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """设置DOF状态张量 - 支持多环境并行"""
        if self._env.agent.robot is None:
            return
            
        # try:
        # 获取当前的 qpos 和 qvel
        current_qpos = self._env.agent.robot.qpos.clone()  # [num_envs, num_dof]
        current_qvel = self._env.agent.robot.qvel.clone()  # [num_envs, num_dof]
        current_qpos[set_env_ids] = dof_states[set_env_ids, :self.num_dof]
        current_qvel[set_env_ids] = dof_states[set_env_ids, self.num_dof:2*self.num_dof]
        
        # 按环境索引修改对应的 qpos 和 qvel
        # for env_id in set_env_ids:
        #     if env_id < len(dof_states) and env_id < current_qpos.shape[0]:
        #         dof_state = dof_states[env_id]
                
        #         # 解析DOF状态
        #         dof_pos = dof_state[:self.num_dof]
        #         dof_vel = dof_state[self.num_dof:2*self.num_dof]
                
        #         # 确保数据类型和设备一致
        #         if hasattr(dof_pos, 'to'):
        #             dof_pos = dof_pos.to(current_qpos.device)
        #         if hasattr(dof_vel, 'to'):
        #             dof_vel = dof_vel.to(current_qvel.device)
                
        #         # 修改对应环境的 qpos 和 qvel
        #         current_qpos[env_id] = dof_pos
        #         current_qvel[env_id] = dof_vel
        
        # 一次性设置所有环境的 qpos 和 qvel
        self._env.agent.robot.set_qpos(current_qpos)
        self._env.agent.robot.set_qvel(current_qvel)
        
        self._env.scene._gpu_apply_all()
        self._env.scene._gpu_fetch_all()
        # except Exception as e:
        #     logger.error(f"Failed to set DOF state: {e}")
        #     import traceback
        #     traceback.print_exc()

    def simulate_at_each_physics_step(self, action=None):
        """在每个物理步骤进行仿真 - 参考IsaacSim的仿真步骤逻辑"""
        # 当action不是全0的时候：
#         robot = env.agent.robot
# # 所有 active 关节的 limit
# qlim = robot.get_qlimits()          # shape (N, 2)  N=总DOF
# lower = qlim[:, 0]
# upper = qlim[:, 1]

        # q_limits = self._env.agent.robot.get_qlimits()  # shape [dof, 2]
        # q_min, q_max = q_limits[0,:,0], q_limits[0,:,1]
        q_min = torch.tensor([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
                             -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, 
                             -2.618, -0.52, -0.52,
                             -3.0892, -1.5882, -2.618, -1.0472, 
                             -3.0892, -2.2515, -2.618, -1.0472]).to(action.device)
        q_max = torch.tensor([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                             2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 
                             2.618, 0.52, 0.52,
                             2.6704, 2.2515, 2.618, 2.0944,
                             2.6704, 1.5882, 2.618, 2.0944]).to(action.device)
        # 裁剪到物理可达范围
        print("q_min: ", q_min)
        print("q_max: ", q_max)
        action = action + torch.tensor(self._env.agent.keyframes["standing"].qpos).to(action.device) 
        # action = torch.zeros_like(action)
        print("action: ",action)
        target_q = torch.clip(action, q_min, q_max)
        # 归一化到 [-1, 1]
        action = 2.0 * (target_q - q_min) / (q_max - q_min + 1e-8) - 1.0
        # action = torch.zeros_like(action)
        # action[:,1] = 1.0
        action = torch.clip(action, -1, 1)
        print("default qpos: ",self._env.agent.keyframes["standing"].qpos)
        # if not torch.all(action == 0):
        #     self.scene._gpu_apply_all()
        #     self._env.step(action+torch.tensor(self._env.agent.keyframes["standing"].qpos, device=action.device))
        #     self.scene._gpu_fetch_all()
        # else:
        print("action norm: ", action)
        self.scene._gpu_apply_all()
        self._env.step(action=action)
        self.scene._gpu_fetch_all()
        # self.scene.step()
        

    def setup_viewer(self):
        """
        设置查看器 - 参考Genesis和IsaacGym的查看器设置逻辑
        创建ManiSkill的查看器并设置键盘事件监听
        """
        if self.headless:
            logger.info("Headless mode enabled, skipping viewer setup")
            return
            
        if self.scene is None:
            logger.warning("Scene not available, cannot setup viewer")
            return
            
        try:
            # 启用查看器同步
            self.enable_viewer_sync = True
            self.visualize_viewer = True
            
            # 检查render_mode配置
            render_mode = getattr(self.simulator_config.sim, 'render_mode', 'human')
            logger.info(f"Render mode: {render_mode}")
            
            if render_mode == "human":
                # 设置ManiSkill场景的查看器
                # if hasattr(self.scene, 'setup_viewer'):
                #     self.scene.setup_viewer()
                #     logger.info("ManiSkill scene viewer setup completed")
                # else:
                #     logger.warning("Scene does not have setup_viewer method")
                
                # # 设置相机控制器（如果可用）
                # if self.viewport_camera_controller is not None:
                #     try:
                #         self.viewport_camera_controller.setup_viewer()
                #         logger.info("Viewport camera controller viewer setup completed")
                #     except Exception as e:
                #         logger.warning(f"Failed to setup viewport camera controller viewer: {e}")
                
                # # 设置查看器引用（ManiSkill的查看器通常通过scene访问）
                # self.viewer = getattr(self.scene, 'viewer', None)
                # if self.viewer is None:
                #     # 尝试从子场景获取查看器
                #     if hasattr(self.scene, 'sub_scenes') and self.scene.sub_scenes:
                #         self.viewer = getattr(self.scene.sub_scenes[0], 'viewer', None)

                self.viewer = self._env.unwrapped.viewer
                
                logger.info("Viewer setup completed successfully")
            else:
                logger.info(f"Render mode '{render_mode}' does not require viewer setup")
            
        except Exception as e:
            logger.error(f"Failed to setup viewer: {e}")
            self.visualize_viewer = False
            self.viewer = None

    def query_viewer_has_closed(self):
        """
        查询查看器是否已关闭 - 参考IsaacGym的实现
        Returns:
            bool: 如果查看器已关闭返回True，否则返回False
        """
        if self.viewer is None:
            return True
            
        try:
            if hasattr(self.viewer, 'is_closed'):
                return self.viewer.is_closed()
            elif hasattr(self.viewer, 'closed'):
                return self.viewer.closed
            else:
                # 如果没有关闭检查方法，假设查看器仍然打开
                return False
        except Exception as e:
            logger.warning(f"Failed to query viewer close status: {e}")
            return False

    def poll_viewer_events(self):
        """
        轮询查看器事件 - 参考IsaacGym的实现
        处理键盘输入和其他查看器事件
        """
        if self.viewer is None or not self.visualize_viewer:
            return
            
        try:
            if hasattr(self.viewer, 'poll_events'):
                self.viewer.poll_events()
            elif hasattr(self.scene, 'poll_viewer_events'):
                self.scene.poll_viewer_events()
        except Exception as e:
            logger.warning(f"Failed to poll viewer events: {e}")

    def set_viewer_camera_look_at(self, camera_pos, camera_target):
        """
        设置查看器相机视角 - 参考IsaacGym的实现
        
        Args:
            camera_pos (list): 相机位置 [x, y, z]
            camera_target (list): 相机目标位置 [x, y, z]
        """
        if self.viewer is None or not self.visualize_viewer:
            return
            
        try:
            if hasattr(self.viewer, 'set_camera_look_at'):
                self.viewer.set_camera_look_at(camera_pos, camera_target)
            elif hasattr(self.scene, 'set_viewer_camera_look_at'):
                self.scene.set_viewer_camera_look_at(camera_pos, camera_target)
            else:
                logger.debug("Viewer does not support camera look at control")
        except Exception as e:
            logger.warning(f"Failed to set viewer camera look at: {e}")

    def render(self, sync_frame_time=True):
        """
        渲染环境 - 参考Genesis和IsaacGym的渲染逻辑
        支持查看器同步和帧时间同步
        """
        if self.headless or not self.visualize_viewer:
            return None
            
        if self.scene is None:
            logger.warning("Scene not available for rendering")
            return None
            
        try:
            # 如果启用了查看器同步，进行同步渲染
            if self.enable_viewer_sync:
                self._env.scene._gpu_apply_all()
                self._env.render()
                self._env.scene._gpu_fetch_all()
            else:
                # 非同步渲染
                if hasattr(self.scene, 'render'):
                    return self.scene.render()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to render: {e}")
            return None

    def step(self, actions):
        """执行仿真步骤 - 参考IsaacSim的步骤执行逻辑"""
        if self._env.agent is None:
            return
            
        try:
            # 应用动作
            if hasattr(self._env.agent, 'step'):
                self._env.agent.step(actions)
            
            # 仿真步骤
            self.simulate_at_each_physics_step()
            
            # 刷新张量
            self.refresh_sim_tensors()
            
        except Exception as e:
            logger.error(f"Failed to step simulation: {e}")

    def reset_envs(self, env_ids):
        for env_id in env_ids:
            default_qpos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            self._env.agent.robot.set_qpos(default_qpos, env_idx=env_id)
        
            base_pos = self.base_init_state[:3]
            base_rot = self.base_init_state[3:7]
            # 转换为numpy数组
            base_pos_tensor = torch.tensor(base_pos, dtype=torch.float, device=self.device) if not isinstance(base_pos, torch.Tensor) else base_pos.to(self.device)
            base_rot_tensor = torch.tensor(base_rot, dtype=torch.float, device=self.device) if not isinstance(base_rot, torch.Tensor) else base_rot.to(self.device)
            self._env.agent.robot.set_pose(sapien.Pose(p=base_pos_tensor, q=base_rot_tensor), env_idx=env_id)
        
        logger.info(f"Reset environments: {env_ids}")


    def get_observations(self):
        """获取观察 - 参考IsaacSim的观察获取逻辑"""
        if self._env.agent.robot is None:
            return {}
            
        try:
            obs = {}
            
            # 获取基座状态
            if hasattr(self._env.agent.robot, 'pose'):
                pose = self._env.agent.robot.pose
                obs['base_pos'] = pose.p
                obs['base_quat'] = pose.q
            
            # 获取关节状态
            if hasattr(self._env.agent.robot, 'get_qpos'):
                obs['dof_pos'] = self._env.agent.robot.get_qpos()
            
            if hasattr(self._env.agent.robot, 'get_qvel'):
                obs['dof_vel'] = self._env.agent.robot.get_qvel()
            
            # 获取刚体状态
            if self._rigid_body_pos is not None:
                obs['rigid_body_pos'] = self._rigid_body_pos
            
            if self._rigid_body_rot is not None:
                obs['rigid_body_rot'] = self._rigid_body_rot
            
            return obs
            
        except Exception as e:
            logger.error(f"Failed to get observations: {e}")
            return {}

    def close(self):
        """关闭仿真器 - 参考Genesis和IsaacGym的关闭逻辑"""
        try:
            # 关闭查看器
            if self.viewer is not None:
                try:
                    if hasattr(self.viewer, 'close'):
                        self.viewer.close()
                    logger.info("Viewer closed")
                except Exception as e:
                    logger.warning(f"Failed to close viewer: {e}")
                finally:
                    self.viewer = None
            
            # 关闭场景
            if self.scene is not None:
                self.scene.close()
            
            # 关闭相机控制器
            if self.viewport_camera_controller is not None:
                self.viewport_camera_controller.close()
            
            # 重置查看器相关状态
            self.visualize_viewer = False
            self.enable_viewer_sync = False
                
            logger.info("ManiSkill simulator closed")
            
        except Exception as e:
            logger.error(f"Failed to close simulator: {e}")

    @property
    def dof_state(self):
        """DOF状态属性 - 参考IsaacSim的dof_state属性"""
        # 确保dof_pos和dof_vel存在
        # self.dof_pos = self._env.agent.robot.qpos
        # self.dof_vel = self._env.agent.robot.qvel
        # 连接位置和速度：dof_pos[..., None] + dof_vel[..., None]
        return torch.cat([self.dof_pos, self.dof_vel], dim=-1)

    # ----- Debug Visualization Methods -----
    
    def add_visualize_entities(self, num_visualize_markers: int, radius: float = 0.04):
        """
        为调试和可视化目的，仅在第一个环境中创建无碰撞的运动学球体标记。
        (ManiSkill 3.0 兼容版本)
        
        Args:
            num_visualize_markers (int): 要创建的可视化标记数量。
            radius (float): 所有标记的球体半径。
        """
        if not hasattr(self, 'scene') or self.scene is None:
            logger.warning("Scene not available, cannot create visualize entities")
            return
            
        self.visualize_entities = []
        
        try:
            for i in range(num_visualize_markers):
                # --- 修改点：只为第一个环境 (env_idx = 0) 创建 ---
                env_idx = 0
                try:
                    target_scene = self.scene
                    if hasattr(self.scene, 'sub_scenes') and self.scene.sub_scenes:
                        target_scene = self.scene.sub_scenes[env_idx]

                    # --- 核心修改点：使用 ManiSkill 的 ActorBuilder ---
                    builder = target_scene.create_actor_builder()
                    
                    # 1. 只添加视觉形状 .add_sphere_visual()，不添加碰撞体 (NO .add_sphere_collision())
                    builder.add_sphere_visual(
                        radius=radius,
                        material=sapien.render.RenderMaterial(
                            base_color=[1.0, 0.0, 0.0, 1.0], # 默认红色
                        )
                    )
                    
                    # 2. 使用 .build_kinematic() 创建一个运动学体。
                    # 它可被 set_kinematic_target 移动，但因没有碰撞体，不会对其他物体产生物理交互。
                    marker_actor = builder.build_kinematic(name=f"visual_marker_{i}_{env_idx}")
                    
                    self.visualize_entities.append([marker_actor])
                    logger.info(f"Successfully created kinematic marker: {marker_actor.get_name()}")
                    
                except Exception as env_e:
                    logger.warning(f"Failed to create marker for marker {i}, env {env_idx}: {env_e}")
                    self.visualize_entities.append([None])

            self.scene._gpu_apply_all()
            self.scene.px.gpu_init()
            self.scene._gpu_fetch_all()
                
        except Exception as e:
            logger.error(f"An unexpected error occurred in add_visualize_entities: {e}")
            self.visualize_entities = [[None] for _ in range(num_visualize_markers)]

    def clear_lines(self):
        """
        清除调试线条对象。
        在 ManiSkill 中，这个功能暂时不实现，因为需要额外的线条绘制支持。
        """
        # TODO: 如果需要线条绘制功能，可以在这里实现
        # 目前 ManiSkill 的线条绘制支持有限
        pass

    def draw_sphere(self, pos, radius, color, env_id, pos_id=0):
        """
        在第一个环境中，更新一个可视化球体标记的位置和颜色。
        (纯 PyTorch 版本，无 numpy 依赖)

        Args:
            pos (torch.Tensor): 球体位置, 形状应为 [3] 或 [1, 3] 等可以 flatten 的形式。
            color (list): 球体颜色 [R, G, B, A]
            pos_id (int): 可视化实体ID
        """
        if not hasattr(self, 'visualize_entities') or not self.visualize_entities or \
        pos_id >= len(self.visualize_entities):
            return
        
        sphere_entity = self.visualize_entities[pos_id][0]
        
        if sphere_entity is None:
            return
            
        try:
            # --- 修改点：移除 numpy，使用 .tolist() ---
            # .tolist() 会将Tensor转换为标准的Python列表，同样会隐式地将数据从GPU移至CPU
            pos_list = pos.flatten().detach().cpu().tolist()
            
            # 使用Python列表来设置Pose
            sphere_entity.set_pose(sapien.Pose(p=pos_list))
            
            # 颜色更新部分不变，因为它不依赖 numpy
            render_body = sphere_entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body:
                render_shape = render_body.get_render_shapes()[0]
                material = render_shape.material
                material.base_color = color

        except Exception as e:
            logger.error(f"Failed to draw sphere at pos_id={pos_id}: {e}")

    def draw_line(self, start_point, end_point, color, env_id):
        """
        在两点之间绘制线条。
        在 ManiSkill 中，线条绘制功能有限，这里暂时不实现。
        
        Args:
            start_point (torch.Tensor): 起始点 [3]
            end_point (torch.Tensor): 结束点 [3]
            color (list): 线条颜色 [R, G, B, A]
            env_id (int): 环境ID
        """
        # TODO: 如果需要线条绘制功能，可以在这里实现
        # ManiSkill 的线条绘制支持有限，可能需要使用其他方法
        logger.debug(f"Line drawing not implemented in ManiSkill (start: {start_point}, end: {end_point}, color: {color}, env: {env_id})")
        pass

    def create_maniskill_env(self):
        """
        创建一个真正的 ManiSkill Env，并把关键引用对齐到本封装：
        - self._env (gym 环境)
        - self.scene
        - self._env.agent / self._env.agent.robot
        """
        import gymnasium as gym
        import mani_skill.envs  # 触发所有内置/自定义环境注册
        # 上面我们已经在本文件注册了 G1Humanoid-Bridge-v0

        # 获取渲染模式配置
        # render_mode = getattr(self.simulator_config.sim, 'render_mode', 'human')
        render_mode = 'human' if not self.headless else 'rgb_array'
        logger.info(f"Creating ManiSkill environment with render_mode: {render_mode}")
        
        # 
        sim_config = SimConfig(
            sim_freq=self.simulator_config.sim.fps,
            control_freq=self.simulator_config.sim.fps // self.simulator_config.sim.control_decimation,
        )
        
        env = gym.make(
            "G1Humanoid-Bridge-v0",
            num_envs=self.num_envs,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            reward_mode=self.reward_mode,
            render_mode=render_mode,  # 设置渲染模式
            robot_uids="unitree_g1",
            # device参数在环境内部处理，不需要传递
            sim_backend="gpu", render_backend="cuda",  # 需要的话解注
            sim_config=sim_config,
        )

        # reset 一次，激活 buffers
        # _obs, _info = env.reset(options=dict(reconfigure=True))
        
        
        # 对齐引用：以后都走 env 的生命周期
        self._env = env
        _obs, _info = self._env.reset(seed=[2022 + i for i in range(self.num_envs)], options=dict(reconfigure=True))
        # self._env.step(action=None)
        self.scene = env.scene
        # self._env.agent = env.agent
        # self._env.agent.robot = env.agent.robot
        
        # 确保场景更新以进行渲染
        if hasattr(self.scene, 'update_render'):
            self.scene.update_render()
        if isinstance(_obs, dict):
            self.base_pos  = _obs.get("base_pos", None)
            self.base_quat = _obs.get("base_quat", None)
            self.dof_pos   = _obs.get("qpos", None)
            self.dof_vel   = _obs.get("qvel", None)
        else:
            logger.info(f"Observation is not a dict, type: {type(_obs)}")
            # 如果_obs不是dict，我们可以从机器人直接获取状态
            if hasattr(self._env.agent.robot, 'root_pose'):
                self.base_pos = self._env.agent.robot.root_pose.p
                self.base_quat = self._env.agent.robot.root_pose.q[:, [1, 2, 3, 0]]
            if hasattr(self._env.agent.robot, 'qpos'):
                self.dof_pos = self._env.agent.robot.qpos
            if hasattr(self._env.agent.robot, 'qvel'):
                self.dof_vel = self._env.agent.robot.qvel

        # 形状校验：现在 qpos 也是 [num_envs, nq]
        assert self._env.agent.robot.root_pose.p.shape[0] == self.num_envs
        assert self._env.agent.robot.qpos.shape[0] == self.num_envs

    def create_envs(self, num_envs, env_origins, base_init_state, env_config=None):
        logger.info(f"Creating {num_envs} ManiSkill environments via bridge env...")
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        # 交给真正的 ManiSkill 环境构建（会自动创建 scene / robot）
        self.create_maniskill_env()
        # self._set_robot_initial_state()

        logger.info(f"Created {num_envs} environments (via G1Humanoid-Bridge-v0)")
        return [self], list(range(num_envs))

@register_env("G1Humanoid-Bridge-v0")
class G1HumanoidBridgeEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ("state",)
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    SUPPORTED_ROBOTS = ["g1_humanoid"]
    
    def __init__(self, *args, **kwargs):
        # 在调用父类初始化之前设置control_mode
        self._control_mode = kwargs.get('control_mode', 'pd_joint_pos')
        self._render_mode = kwargs.get('render_mode', 'human')
        logger.info(f"G1HumanoidBridgeEnv initialized with render_mode: {self._render_mode}")
        super().__init__(*args, **kwargs)
    
    @property
    def control_mode(self):
        return self._control_mode

    def _load_scene(self, options: dict):
        from mani_skill.utils.building.ground import build_ground
        build_ground(self.scene)  # 铺地面，够用了
        
        # 添加光照设置，确保场景有足够的光照
        # self.scene.set_ambient_light([0.3, 0.3, 0.3])
        # self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate=False):
        # 复用你的自定义 G1 机器人
        from humanoidverse.simulator.maniskill.g1_robot import G1HumanoidAgent
        self.agent = G1HumanoidAgent(
            scene=self.scene,
            control_mode=self.control_mode,
            control_freq=self._control_freq,
            initial_pose=None,
        )
        # print(self.agent.robot.joints[0].damping)
        if initial_agent_poses is None:
            initial_agent_poses = sapien.Pose(p=[0, 0, 1.0])
        # 关键：分子场景构建，让 qpos 也带 batch 维 [num_envs, nq]
        # 确保在调用super之前agent已经被设置
        assert self.agent is not None, "Agent must be set before calling super()._load_agent"
        
        # 手动设置agents列表，避免MultiAgent初始化问题
        self.agents = [self.agent]
        
        
        # 不再调用super()._load_agent，因为我们已经手动设置了agent
        # super()._load_agent(options, initial_agent_poses=initial_agent_poses, build_separate=True)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        nq = self.agent.robot.qpos.shape[-1]
        # self.agent.robot.set_qpos(torch.zeros((env_idx.numel(), nq), device=self.device))
        
        # 确保机器人处于站立姿态
        if hasattr(self.agent, 'keyframes') and 'standing' in self.agent.keyframes:
            standing_keyframe = self.agent.keyframes['standing']
            self.agent.robot.set_qpos(standing_keyframe.qpos)
            self.agent.robot.set_root_pose(standing_keyframe.pose)

    def _get_obs(self):
        if self.obs_mode == "state":
            r = self.agent.robot
            return dict(
                base_pos=r.root_pose.p,   # [N,3]
                base_quat=r.root_pose.q,  # [N,4]
                qpos=r.qpos,              # [N,nq]
                qvel=r.qvel,              # [N,nq]
            )
        return super()._get_obs()

    def compute_dense_reward(self, obs=None, action=None, info=None):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_sparse_reward(self, obs=None, action=None, info=None):
        return (self.compute_dense_reward() > 0).float()

    def evaluate(self):
        return dict(success=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
    
    # def render(self):
    #     """渲染环境"""
    #     if self._render_mode == "human":
    #         # 如果有viewer，进行渲染
    #         if hasattr(self, 'viewer') and self.viewer is not None:
    #             # return self.viewer.render()
    #             return self._env.render()
    #         else:
    #             # 尝试通过scene的viewer进行渲染
    #             if hasattr(self.scene, 'viewer') and self.scene.viewer is not None:
    #                 return self.scene.viewer.render()
    #             else:
    #                 logger.debug("No viewer available for rendering")
    #                 return None
    #     else:
    #         logger.debug(f"Render mode '{self._render_mode}' does not support rendering")
    #         return None