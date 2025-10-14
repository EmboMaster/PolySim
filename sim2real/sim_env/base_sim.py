import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
import threading
import numpy as np
import time
from loguru import logger
import argparse
import yaml
from threading import Thread
import sys
from std_msgs.msg import Float64MultiArray, Bool
sys.path.append('../')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

from std_msgs.msg import Float64MultiArray
from sim2real.utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

class BaseSimulator:
    def __init__(self, config, node):
        self.config = config
        self.node = node
        self.rate = self.node.create_rate(1/self.config["SIMULATE_DT"])
        self.viewer_rate = self.node.create_rate(1/self.config["VIEWER_DT"])
        self.init_config()
        self.init_scene()
        self.init_unitree_bridge()

        # for more scenes
        self.init_subscriber()
        self.init_publisher()
        
        self.sim_thread = Thread(target=self.SimulationThread)

    def init_subscriber(self):
        self.disable_elastic_band_subscriber = self.node.create_subscription(
            Bool,
            '/disable_elastic_band',
            self.disable_elastic_band_callback,
            10
        )

    def init_publisher(self):
        pass

    def disable_elastic_band_callback(self, msg):
        if msg.data and self.elastic_band_active:
            self.elastic_band_active = False
            logger.info("Received disable signal - Elastic band disabled")
        if not(msg.data) and not(self.elastic_band_active):
            self.elastic_band_active = True
            logger.info("Received disable signal - Elastic band disabled")

    def init_config(self):
        self.robot = Robot(self.config)
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.elastic_band_active = True

    def print_fixed_properties(self, verbose: bool = True):
        """
        一次性打印所有固定物理常量（质量、惯量、关节极限、执行器范围等）。
        结果只依赖 self.mj_model，运行一次即可。
        """
        m = self.mj_model
        if verbose:
            print("\n========== 固定物理特性 ==========")

        # 1. DOF / 关节
        print("\n---- DOF/关节 ----")
        for j in range(m.njnt):
            dof_id = m.jnt_dofadr[j]
            name   = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            rng    = m.jnt_range[j]
            damp   = m.dof_damping[dof_id]
            arma   = m.dof_armature[dof_id]
            fric   = m.dof_frictionloss[dof_id]
            print(f"{name:25s}  dof={dof_id:2d}  range=[{rng[0]:7.3f}, {rng[1]:7.3f}]  "
                f"damp={damp:6.3f}  armature={arma:6.3f}  friction={fric:6.3f}")

        # 2. 执行器
        print("\n---- 执行器 ----")
        for a in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator_{a}"
            joint_id = m.actuator_trnid[a, 0]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            ctrl_rng = m.actuator_ctrlrange[a]
            print(f"{name:25s}  joint={jname:25s}  ctrl=[{ctrl_rng[0]:7.2f}, {ctrl_rng[1]:7.2f}]")

        # 3. 刚体（质量 + 主惯量）
        print("\n---- 刚体 ----")
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            mass = m.body_mass[b]
            I    = m.body_inertia[b]
            print(f"{name:25s}  mass={mass:7.3f}  I={np.array2string(I, precision=4)}")

        if verbose:
            print("\n========== 打印完毕 ==========")

    def init_scene(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        # Enable the elastic band
        self.print_fixed_properties()
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "h1" in self.config["ROBOT_TYPE"] or "g1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.mj_model, self.mj_data, self.config)
        # if self.config["PRINT_SCENE_INFORMATION"]:
        #     self.unitree_bridge.PrintSceneInformation()
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"])

    def compute_torques(self):
        if self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_motor):
                if self.unitree_bridge.use_sensor:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.num_motor]
                        )
                    )
                else:
                    self.torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.qpos[7+i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[6+i]
                        )
                    )
                    print("self.unitree_bridge.low_cmd.motor_cmd[i].tau: ",self.unitree_bridge.low_cmd.motor_cmd[i].tau)
                    print("self.unitree_bridge.low_cmd.motor_cmd[i].kp: ",self.unitree_bridge.low_cmd.motor_cmd[i].kp)
                    print("self.unitree_bridge.low_cmd.motor_cmd[i].kd: ",self.unitree_bridge.low_cmd.motor_cmd[i].kd)
                    
        # Set the torque limit
        self.torques = np.clip(self.torques, 
                               -self.unitree_bridge.torque_limit, 
                               self.unitree_bridge.torque_limit)

    def sim_step(self):
        self.unitree_bridge.PublishLowState()
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        # if self.config["ENABLE_ELASTIC_BAND"]:
        #     if self.elastic_band.enable:
        #         self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
        #             self.mj_data.qpos[:3], self.mj_data.qvel[:3]
        #         )
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable and self.elastic_band_active:
                # 计算弹性带的力
                elastic_force = self.elastic_band.Advance(self.mj_data.qpos[:3], self.mj_data.qvel[:3])
                
                # 对力进行缩放，减小其强度
                scaling_factor = 0.6  # 设置缩放因子
                elastic_force *= scaling_factor
                
                # 设置弹性带施加的力
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = elastic_force

        self.compute_torques()
        if self.unitree_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else: 
            # print("self.torques: ",self.torques.shape)
            # print("self.torques: ",self.torques)
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def SimulationThread(self,):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            # self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.node.get_logger().info(f"FPS: {100 / (end_time - start_time)}")
                start_time = end_time
            self.rate.sleep()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])
    rclpy.init(args=None)
    node = rclpy.create_node('sim_mujoco')

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    # print("config: ",config)
    simulation = BaseSimulator(config, node)
    simulation.sim_thread.start()