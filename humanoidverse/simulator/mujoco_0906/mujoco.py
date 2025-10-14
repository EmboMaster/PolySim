import sys
import os
from loguru import logger
import torch
import numpy as np
from termcolor import colored
from rich.progress import Progress
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
import copy
import time
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

class MuJoCo(BaseSimulator):
    """
    Pure MuJoCo-based robotic simulation environment, providing a framework for simulation setup, 
    environment creation, and control over robotic assets and simulation properties.
    """
    def __init__(self, config, device):
        """
        Initializes the MuJoCo simulator with configuration settings and simulation device.
        
        Args:
            config (dict): Configuration dictionary for the simulation.
            device (str): Device type for simulation ('cpu' or 'cuda').
        """
        self.cfg = config
        self.sim_cfg = config.simulator.config
        self.robot_cfg = config.robot
        self.device = device
        self.sim_device = device
        self.headless = False
        
        # MuJoCo specific initialization
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        self.viewer_dt = 0.02
    # ----- Configuration Setup Methods -----

    def set_headless(self, headless):
        """
        Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without graphical display.
        """
        self.headless = headless

    def setup(self):
        """
        Initializes the simulator parameters and environment.
        """
        self.sim_dt = 1 / self.sim_cfg.sim.fps
        self.sim_substeps = self.sim_cfg.sim.substeps
        
        # Setup will be completed when assets are loaded
        logger.info(f"MuJoCo simulator setup with dt={self.sim_dt}")

    # ----- Terrain Setup Methods -----

    def setup_terrain(self, mesh_type):
        """
        Configures the terrain based on specified mesh type. 
        In MuJoCo, terrain is typically defined in the XML file.

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'heightfield', 'trimesh').
        """
        if mesh_type == 'plane':
            logger.info("Plane terrain will be loaded from XML file")
        elif mesh_type == 'trimesh':
            raise NotImplementedError(f"Mesh type {mesh_type} hasn't been implemented in MuJoCo subclass.")
        else:
            logger.warning(f"Terrain type {mesh_type} not explicitly handled, assuming it's in XML")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self):
        """
        Loads the robot assets into the simulation environment using MuJoCo.
        """
        # Get initial state
        init_quat_xyzw = self.robot_cfg.init_state.rot
        init_quat_wxyz = init_quat_xyzw[-1:] + init_quat_xyzw[:3]
        self.base_init_pos = torch.tensor(
            self.robot_cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            init_quat_wxyz, device=self.device
        )

        # Load MuJoCo model
        asset_root = self.robot_cfg.asset.asset_root
        if hasattr(self.robot_cfg.asset, 'xml_file'):
            asset_file = self.robot_cfg.asset.xml_file
            #print(asset_file)
        else:
            #Fallback to a default or raise error
            asset_file = 'g1/g1_29dof_anneal_23dof.xml'  # Default from your example
            
        asset_path = os.path.join(asset_root, asset_file)
        
        logger.info(f"Loading MuJoCo model from: {asset_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        # print("qpos: ", self.mj_data.qpos.shape)
        # print("qvel: ", self.mj_data.qvel.shape)
        # print("xpos: ", self.mj_data.xpos.shape)
        # print("xquat: ", self.mj_data.xquat.shape)
        self.asset_path = asset_path
        # Setup viewer if not headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=True
            )

        # Get DOF information
        dof_names_list = copy.deepcopy(self.robot_cfg.dof_names)
        
        # Find DOF indices in MuJoCo
        # self.dof_ids = [6,..., 28]
        self.dof_ids = []
        for name in dof_names_list:
            try:
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    dof_adr = self.mj_model.jnt_dofadr[joint_id]
                    self.dof_ids.append(dof_adr)
                else:
                    logger.warning(f"Joint {name} not found in MuJoCo model")
            except Exception as e:
                logger.error(f"Error finding joint {name}: {e}")
        print("dof_ids: ", self.dof_ids)
        # Get body information
        self.body_names = self.robot_cfg.body_names
        # print("body_names: ", self.body_names)
        self.num_bodies = len(self.body_names)
        self.dof_names = dof_names_list
        #print("dof_names: ", self.dof_names)
        self.num_dof = len(self.dof_ids)

        # for name in dof_names_list:
        #     joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        #     if joint_id >= 0:
        #         dof_adr = self.mj_model.jnt_dofadr[joint_id]
        #         # 打印关节与控制输入的对应关系
        #         print(f"Joint {name} has control index at {dof_adr}")
        #         # 这里假设每个关节有一个自由度，实际上可能有多个自由度
        #         print(f"Control value for {name}: {self.mj_data.ctrl[dof_adr]}")
        #     else:
        #         print(f"Joint {name} not found in the model")
        # Map body names to indices
        # self._body_list = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 
        # 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 
        # 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 
        # 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
        # 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 
        # 'right_elbow_link']
        self.body_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                 for name in self.body_names]
        # print("body_ids: ",len(self.body_ids))
        # print("body_ids: ",self.body_ids)
        # body_ids:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27]

        # Create _body_list for compatibility (similar to Genesis version)
        self._body_list = []
        for i in self.body_ids:
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            #print(body_name)
            if body_name:
                self._body_list.append(body_name)
            else:
                self._body_list.append(f"body_{i}")  # fallback name
        # print("_body_list: ", self._body_list)
        # print("_body_list: ",len(self._body_list))
        logger.info(f"Loaded robot with {self.num_dof} DOFs and {self.num_bodies} bodies")
        logger.info(f"Total bodies in model: {len(self._body_list)}")

    # ----- Environment Creation Methods -----

    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body, or None if not found.
        """
        try:
            return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except:
            return None

    def create_envs(self, num_envs, env_origins, base_init_state):
        """
        Creates and initializes environments with specified configurations.
        Note: MuJoCo typically handles single environment. For multiple environments,
        you would need multiple MuJoCo instances.

        Args:
            num_envs (int): Number of environments to create.
            env_origins (list): List of origin positions for each environment.
            base_init_state (array): Initial state of the base.
        """
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        
        if num_envs > 1:
            logger.warning("MuJoCo typically handles single environment. Multiple environments not fully implemented.")
        
        # Reset to initial state
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        
        # Set initial position and orientation
        if self.mj_data.qpos.shape[0] >= 7:  # Assuming first 7 are root position and quaternion
            self.mj_data.qpos[:3] = self.base_init_pos.cpu().numpy()
            self.mj_data.qpos[3:7] = self.base_init_quat.cpu().numpy()  # wxyz format
        
        return None, None

    def get_dof_limits_properties(self):
        """
        Retrieves the DOF (degrees of freedom) limits and properties from MuJoCo model.
        
        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        
        for i, dof_idx in enumerate(self.dof_ids):
            # Get joint info
            joint_id = None
            for j in range(self.mj_model.njnt):
                if self.mj_model.jnt_dofadr[j] <= dof_idx < self.mj_model.jnt_dofadr[j] + self.mj_model.jnt_type[j]:
                    joint_id = j
                    break
            
            if joint_id is not None:
                # Position limits
                pos_range = self.mj_model.jnt_range[joint_id]
                self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
                self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
                self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
                self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
                
                # Apply soft limits if configured
                
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                soft_limit = self.cfg.rewards.reward_limit.soft_dof_pos_limit
                self.dof_pos_limits[i, 0] = m - 0.5 * r * soft_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * soft_limit
                
                # Get actuator limits for this joint
                for act_id in range(self.mj_model.nu):
                    if self.mj_model.actuator_trnid[act_id, 0] == joint_id:
                        # Torque limits
                        self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
                        #print("torque_limits: ", self.torque_limits[i])
                        break
            else:
                # Fallback to config values if available
                if hasattr(self.robot_cfg, 'dof_pos_lower_limit_list'):
                    self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
                    self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
                    self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
                    self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
                    self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
                    self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
        
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        Prepares the simulation environment and refreshes any relevant tensors.
        """
        # Forward step to compute initial state
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # Initialize state tensors
        self.refresh_sim_tensors()

    def refresh_sim_tensors(self):
        """
        Refreshes the state tensors in the simulation to ensure they are up-to-date.
        """
        def to_torch(x):
            """Convert numpy array to torch tensor"""
            return torch.tensor(x[None] if x.ndim == 1 else x, device=self.device, dtype=torch.float)

        # Root state (base position, orientation, velocities)
        if self.mj_data.qpos.shape[0] >= 7:
            self.base_pos = to_torch(self.mj_data.qpos[:3])  # Position (x, y, z)
            base_quat_wxyz = to_torch(self.mj_data.qpos[3:7])  # Quaternion (w, x, y, z)
            # Convert from wxyz to xyzw
            self.base_quat = base_quat_wxyz[..., [1, 2, 3, 0]]  # xyzw format
        else:
            self.base_pos = torch.zeros(1, 3, device=self.device)
            self.base_quat = torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float)

        if self.mj_data.qvel.shape[0] >= 6:
            # Transform velocities to body frame
            base_quat_wxyz = self.mj_data.qpos[3:7] if self.mj_data.qpos.shape[0] >= 7 else [1, 0, 0, 0]
            rot_matrix = R.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]).as_matrix()
            
            world_lin_vel = self.mj_data.qvel[:3]  # Linear velocity (x, y, z)
            world_ang_vel = self.mj_data.qvel[3:6]  # Angular velocity (x, y, z)
            
            # Transform to body frame
            body_lin_vel = rot_matrix.T @ world_lin_vel
            body_ang_vel = rot_matrix.T @ world_ang_vel
            
            self.base_lin_vel = to_torch(world_lin_vel)
            self.base_ang_vel = to_torch(world_ang_vel)
        else:
            self.base_lin_vel = torch.zeros(1, 3, device=self.device)
            self.base_ang_vel = torch.zeros(1, 3, device=self.device)

        # Combine root states
        self.all_root_states = torch.cat([
            self.base_pos,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
        ], dim=-1)
        self.robot_root_states = self.all_root_states

        # DOF states
        if len(self.dof_ids) > 0:
            dof_pos_np = np.array(self.mj_data.qpos[7:])
            dof_vel_np = np.array(self.mj_data.qvel[6:])
            self.dof_pos = to_torch(dof_pos_np)
            self.dof_vel = to_torch(dof_vel_np)
        else:
            self.dof_pos = torch.zeros(1, self.num_dof, device=self.device)
            self.dof_vel = torch.zeros(1, self.num_dof, device=self.device)

        contact_forces = self.mj_data.cfrc_ext[self.body_ids, :3]  # (24, 3)
        self.contact_forces = torch.tensor(contact_forces, device=self.device, dtype=torch.float)[None]
        print("contact_force_shape", self.contact_forces.shape)
        # Rigid body states
        if len(self.body_ids) > 0:
            body_positions = []
            body_orientations = []
            body_velocities = []
            body_ang_velocities = []
            
            for body_id in self.body_ids:
                # Position (x, y, z)
                pos = self.mj_data.xpos[body_id]
                body_positions.append(pos)
                
                # Orientation (quaternion in wxyz, convert to xyzw)
                quat_wxyz = self.mj_data.xquat[body_id]
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                body_orientations.append(quat_xyzw)
                
                # Velocities
                lin_vel = self.mj_data.cvel[body_id][:3]
                ang_vel = self.mj_data.cvel[body_id][3:]  
                body_velocities.append(lin_vel)
                body_ang_velocities.append(ang_vel)
            
            self._rigid_body_pos = torch.tensor(body_positions, device=self.device, dtype=torch.float)[None]
            self._rigid_body_rot = torch.tensor(body_orientations, device=self.device, dtype=torch.float)[None]
            self._rigid_body_vel = torch.tensor(body_velocities, device=self.device, dtype=torch.float)[None]
            self._rigid_body_ang_vel = torch.tensor(body_ang_velocities, device=self.device, dtype=torch.float)[None]
        else:
            self._rigid_body_pos = torch.zeros(1, self.num_bodies, 3, device=self.device)
            self._rigid_body_rot = torch.zeros(1, self.num_bodies, 4, device=self.device)
            self._rigid_body_vel = torch.zeros(1, self.num_bodies, 3, device=self.device)
            self._rigid_body_ang_vel = torch.zeros(1, self.num_bodies, 3, device=self.device)

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        Applies the specified torques to the robot's degrees of freedom (DOF).

        Args:
            torques (tensor): Tensor containing torques to apply.
        """
        # 遍历所有自由度（DOF）对应的控制输入
        # for i in range(self.mj_model.nu):  # self.mj_model.nu 是自由度的数量
        #     # 获取控制输入对应的关节 ID
        #     joint_id = self.mj_model.dof_jntid[i]  # 获取自由度对应的关节 ID
        #     joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)  # 根据 ID 获取关节名称
        #     control_input = self.mj_data.ctrl[i]  # 获取该关节的控制输入（如力矩）

        #     print(f"Control input {i} corresponds to joint {joint_name} with control value: {control_input}")
        if isinstance(torques, torch.Tensor):
            torques_np = torques.cpu().numpy().flatten()
        else:
            torques_np = np.array(torques).flatten()

        # for i, dof_name in enumerate(self.dof_names):
        #     joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
        #     if joint_id >= 0:
        #         print(f"Joint {dof_name} with ID {joint_id} corresponds to control input {torques_np[i]}")
        #     else:
        #         print(f"Joint {dof_name} not found in model!")

        # Apply torques to MuJoCo control

        if len(torques_np) <= self.mj_model.nu:
            self.mj_data.ctrl[:] = torques_np[:] 
            #print("mj_data.ctrl = ",self.mj_data.ctrl)
        else:
            logger.warning(f"Too many torques ({len(torques_np)}) for available actuators ({self.mj_model.nu})")

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """
        Sets the root state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            root_states (tensor): New root states to apply.
        """
        #print("set_actor_root_state_tensor")
        #import ipdb; ipdb.set_trace()
        # root_states = torch.cat(
        #     [
        #         self.base_pos,
        #         self.base_quat,
        #         self.base_lin_vel,
        #         self.base_ang_vel,
        #     ], dim=-1
        # )
        root_states = self.robot_root_states
        print("base_ang_vel: ",self.base_ang_vel)
        if isinstance(set_env_ids, torch.Tensor):
            env_ids = set_env_ids.cpu().numpy()
        else:
            env_ids = np.array(set_env_ids)
            
        if isinstance(root_states, torch.Tensor):
            states = root_states.cpu().numpy()
        else:
            states = np.array(root_states)
        
        # For single environment, just use the first state
        if len(states.shape) > 1:
            state = states[0] if len(states) > 0 else states
        else:
            state = states
            
        # Set position and orientation
        if len(state) >= 7 and self.mj_data.qpos.shape[0] >= 7:
            self.mj_data.qpos[:3] = state[:3]  # position
            print("qpos: ",self.mj_data.qpos[:3])
            # Convert from xyzw to wxyz
            self.mj_data.qpos[3:7] = [state[6], state[3], state[4], state[5]]  # orientation
            #self.mj_data.qpos[3:7] = state[3:7]
            print("wxyz: ",self.mj_data.qpos[3:7])
        # Set velocities
        if len(state) >= 13 and self.mj_data.qvel.shape[0] >= 6:
            self.mj_data.qvel[:3] = state[7:10]   # linear velocity
            self.mj_data.qvel[3:6] = state[10:13] # angular velocity
            print("vel: ", self.mj_data.qvel[:6])
            
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """
        Sets the DOF state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            dof_states (tensor): New DOF states to apply.
        """
        if isinstance(dof_states, torch.Tensor):
            states = dof_states.cpu().numpy()
        else:
            states = np.array(dof_states)
            
        # Reshape to get position and velocity
        if len(states.shape) == 3:  # (num_envs, num_dofs, 2)
            states = states[0]  # Take first environment
            
        if len(states.shape) == 2 and states.shape[1] == 2:
            dof_pos = states[:, 0]
            dof_vel = states[:, 1]
        else:
            logger.error(f"Unexpected dof_states shape: {states.shape}")
            return
            
        # Set DOF positions and velocities
        print("reset dof pos: ",dof_pos)
        print("reset dof vel: ",dof_vel)
        for i, dof_idx in enumerate(self.dof_ids):
            if i < len(dof_pos):
                self.mj_data.qpos[dof_idx + 1] = dof_pos[i]
            if i < len(dof_vel):
                self.mj_data.qvel[dof_idx] = dof_vel[i]

    def simulate_at_each_physics_step(self):
        """
        Advances the simulation by a single physics step.
        """
        print("self.mj_data qpos: ",self.mj_data.qpos)
        print("self.mj_data qvel: ",self.mj_data.qvel)
        print("self.mj_data ctrl: ",self.mj_data.ctrl)
        time.sleep(self.viewer_dt)
        mujoco.mj_step(self.mj_model, self.mj_data)
        # Update viewer if available
        if self.viewer is not None:
            self.viewer.sync()
        import ipdb; ipdb.set_trace()
        
    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        if self.mj_model is not None:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=True
                )

    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        if self.viewer is not None:
            self.viewer.sync()
            
    @property
    def dof_state(self):
        """
        Returns the current DOF state (position and velocity).
        """
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def add_visualize_entities(self, num_visualize_markers):
        """
        Adds visualization markers to the scene.
        Note: In MuJoCo, this would typically require modifying the XML or using sites.
        """
        logger.warning("Visualization markers not implemented for pure MuJoCo version")
        self.visualize_entities = []

    # Debug visualization methods
    def clear_lines(self):
        """Clear debug lines (not implemented in basic MuJoCo version)"""
        pass

    def draw_sphere(self, pos, radius, color, env_id, pos_id=0):
        """Draw debug sphere (not implemented in basic MuJoCo version)"""
        pass

    def draw_line(self, start_point, end_point, color, env_id):
        """Draw debug line (not implemented in basic MuJoCo version)"""
        pass
