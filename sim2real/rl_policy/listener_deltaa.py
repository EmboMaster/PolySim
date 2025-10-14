import time
import sys
import signal
import numpy as np
from collections import deque
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import threading
from pynput import keyboard
import os
import argparse
import yaml
sys.path.append('./rl_policy/..')
from utils.robot import Robot

class DataLogger(Node):
    def __init__(self, save_path='data_log.npz', config=None, log_interval=1.0, buffer_rate=50.0, buffer_size=50*600, num_actions=29):
        """
        Initializes the DataLogger node.

        Args:
            save_path (str): Path to save the .npz log file.
            log_interval (float): Interval in seconds for throttled logging.
            buffer_rate (float): Frequency in Hz to update the observation buffer.
            buffer_size (int): Maximum number of entries to store in the buffer.
        """
        super().__init__('data_logger')

        self.config = config
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ 
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported yet")
        self.robot = Robot(config)
        self.num_actions = self.robot.NUM_JOINTS
        
        self.start_recording = False
        self.motion_episode_cnt = 0
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()

        # Initialize buffer for high-frequency data (200 Hz)
        self.obs_buffer = deque(maxlen=buffer_size)  # Buffer size is 200*300 = 60000

        # Initialize subscribers for mocap data
        self.create_subscription(Odometry, "/odometry", self.mocap_callback, 10)

        # Initialize subscribers for Unitree data
        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_state.Init(self.LowStateHandler, 10)
                
        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_cmd.Init(self.LowCmdHandler, 10)

        # Set up the signal handler
        signal.signal(signal.SIGINT, self.sigINT_handler)

        # Throttled logging setup
        self.log_interval = log_interval  # seconds
        self.last_log_time = time.time()
        self.log_timer = self.create_timer(0.1, self.throttled_log_callback)  # High-frequency timer for checking log interval

        # Buffer update timer at 200 Hz
        self.buffer_rate = buffer_rate  # Hz
        self.buffer_timer = self.create_timer(1.0 / self.buffer_rate, self.buffer_update_callback)

        self.get_logger().info("DataLogger node initialized.")

        # Initialize latest data holders
        self.latest_low_state = None
        self.latest_low_cmd = None
        self.latest_mocap = None
        
        # Save path for the .npz file
        self.save_path = save_path

    def clear_buffer(self):
        """
        Clears the observation buffer.
        """
        self.obs_buffer.clear()
        self.get_logger().info("Buffer cleared.")

    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(key):
            try:
                if key.char == ";":

                    self.start_recording = True
                    self.clear_buffer()
                    self.get_logger().info("Start recording for episode {}".format(self.motion_episode_cnt))
                elif key.char == "'":
                    self.start_recording = False
                    self.get_logger().info("Stop recording for episode {}".format(self.motion_episode_cnt))
                    self.process_and_save_data()
                    self.clear_buffer()
                    self.motion_episode_cnt += 1
            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

    # Handler for Unitree low state messages
    def LowStateHandler(self, msg):
        """
        Handles incoming LowState_ messages and stores the latest data.
        """
        timestamp = time.time()
        self.latest_low_state = {
            'timestamp': timestamp,
            'motor_positions': np.array([motor.q for motor in msg.motor_state[:self.num_actions]]),
            'motor_velocities': np.array([motor.dq for motor in msg.motor_state[:self.num_actions]]),
            'motor_torques': np.array([motor.tau_est for motor in msg.motor_state[:self.num_actions]]),
            'imu_quaternion': np.array(msg.imu_state.quaternion),
            'imu_gyroscope': np.array(msg.imu_state.gyroscope),
            'imu_accelerometer': np.array(msg.imu_state.accelerometer)
        }

    # Handler for Unitree low command messages
    def LowCmdHandler(self, msg):
        """
        Handles incoming LowCmd_ messages and stores the latest data.
        """
        timestamp = time.time()
        self.latest_low_cmd = {
            'timestamp': timestamp,
            'motor_commands': {
                "q": np.array([motor_cmd.q for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "dq": np.array([motor_cmd.dq for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "kp": np.array([motor_cmd.kp for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "kd": np.array([motor_cmd.kd for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "tau": np.array([motor_cmd.tau for motor_cmd in msg.motor_cmd[:self.num_actions]])
            }
        }
        

    # Callback function for mocap data
    def mocap_callback(self, msg):
        """
        Handles incoming Mocap data and stores the latest data.
        """


        timestamp = time.time()
        self.latest_mocap = {
            'timestamp': timestamp,
            'position': [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ],
            'orientation': [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z
            ],
            'linear_velocity': [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ],
            'angular_velocity': [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
        }

    # Throttled log callback
    def throttled_log_callback(self):
        """
        Logs the status at specified intervals to avoid excessive logging.
        """
        current_time = time.time()
        if (current_time - self.last_log_time) >= self.log_interval:
            buffer_length = len(self.obs_buffer)
            self.get_logger().info(
                f"Running... Buffer Size: {buffer_length}"
            )
            self.last_log_time = current_time

    # Buffer update callback at 200 Hz
    def buffer_update_callback(self):
        """
        Updates the observation buffer with the latest synchronized data.
        This function runs at a high frequency (e.g., 200 Hz).
        """
        # Fetch the latest data from each category if available

        if self.latest_low_state and self.latest_low_cmd and self.latest_mocap:
            obs = {
                'timestamp': time.time(),
                'low_state': self.latest_low_state,
                'low_cmd': self.latest_low_cmd,
                'mocap': self.latest_mocap
            }
            self.obs_buffer.append(obs)
        else:
            # print(self.latest_low_cmd,self.latest_low_state,self.latest_mocap)
            # If any data is missing, log a warning (throttled)
            # print which data is missing
            if not self.latest_low_state:
                self.get_logger().warn("Missing low state data for buffer update.", throttle_duration_sec=1.0)
            if not self.latest_low_cmd:
                self.get_logger().warn("Missing low command data for buffer update.", throttle_duration_sec=1.0)
            if not self.latest_mocap:
                self.get_logger().warn("Missing mocap data for buffer update.", throttle_duration_sec=1.0)
            self.get_logger().warn("Incomplete data for buffer update.", throttle_duration_sec=1.0)

    # Signal handler to save data upon interruption
    def sigINT_handler(self, sig, frame):
        """
        Handles SIGINT (e.g., Ctrl+C) to gracefully save data and exit.
        """
        # self.get_logger().info("SIGINT received. Saving data...")
        # print("Saving data to ", self.save_path)
        # print("Buffer samples: ", len(self.obs_buffer))
        
        # # Process and save all collected data
        # self.process_and_save_data()
        
        # self.get_logger().info("Data saved successfully for episode {}".format(self.motion_episode_cnt))
        self.get_logger().info("Exiting..., total episodes: {}".format(self.motion_episode_cnt))
        self.get_logger().info("Data saved to {}".format(self.save_path))
        sys.exit(0)

    def process_and_save_data(self):
        """
        Processes the observation buffer and saves it into a single .npz file.
        """
        # Initialize lists for processed data
        buffer_timestamps = []
        buffer_low_state_motor_positions = []
        buffer_low_state_motor_velocities = []
        buffer_low_state_motor_torques = []
        buffer_low_state_imu_quaternion = []
        buffer_low_state_imu_gyroscope = []
        buffer_low_state_imu_accelerometer = []
        # buffer_low_state_foot_force = []
        # buffer_low_state_foot_force_est = []

        buffer_low_cmd_q = []
        buffer_low_cmd_dq = []
        buffer_low_cmd_kp = []
        buffer_low_cmd_kd = []
        buffer_low_cmd_tau = []

        buffer_mocap_positions = []
        buffer_mocap_orientations = []
        buffer_mocap_linear_velocities = []
        buffer_mocap_angular_velocities = []

        # Iterate through the buffer and extract data
        for obs in self.obs_buffer:
            buffer_timestamps.append(obs['timestamp'])

            # Low State
            low_state = obs['low_state']
            buffer_low_state_motor_positions.append(low_state['motor_positions'])
            buffer_low_state_motor_velocities.append(low_state['motor_velocities'])
            buffer_low_state_motor_torques.append(low_state['motor_torques'])
            
            buffer_low_state_imu_quaternion.append(low_state['imu_quaternion'])
            buffer_low_state_imu_gyroscope.append(low_state['imu_gyroscope'])
            buffer_low_state_imu_accelerometer.append(low_state['imu_accelerometer'])
            # buffer_low_state_foot_force.append(low_state['foot_force'])
            # buffer_low_state_foot_force_est.append(low_state['foot_force_est'])

            # Low Command
            low_cmd = obs['low_cmd']
            buffer_low_cmd_q.append(low_cmd['motor_commands']['q'])
            buffer_low_cmd_dq.append(low_cmd['motor_commands']['dq'])
            buffer_low_cmd_kp.append(low_cmd['motor_commands']['kp'])
            buffer_low_cmd_kd.append(low_cmd['motor_commands']['kd'])
            buffer_low_cmd_tau.append(low_cmd['motor_commands']['tau'])

            # Mocap
            mocap = obs['mocap']
            buffer_mocap_positions.append(mocap['position'])
            buffer_mocap_orientations.append(mocap['orientation'])
            buffer_mocap_linear_velocities.append(mocap['linear_velocity'])
            buffer_mocap_angular_velocities.append(mocap['angular_velocity'])

        # Convert lists to NumPy arrays
        buffer_timestamps = np.array(buffer_timestamps)

        buffer_low_state_motor_positions = np.array(buffer_low_state_motor_positions)
        buffer_low_state_motor_velocities = np.array(buffer_low_state_motor_velocities)
        buffer_low_state_motor_torques = np.array(buffer_low_state_motor_torques)
        buffer_low_state_imu_quaternion = np.array(buffer_low_state_imu_quaternion)
        buffer_low_state_imu_gyroscope = np.array(buffer_low_state_imu_gyroscope)
        buffer_low_state_imu_accelerometer = np.array(buffer_low_state_imu_accelerometer)
        # buffer_low_state_foot_force = np.array(buffer_low_state_foot_force)
        # buffer_low_state_foot_force_est = np.array(buffer_low_state_foot_force_est)

        buffer_low_cmd_q = np.array(buffer_low_cmd_q)
        buffer_low_cmd_dq = np.array(buffer_low_cmd_dq)
        buffer_low_cmd_kp = np.array(buffer_low_cmd_kp)
        buffer_low_cmd_kd = np.array(buffer_low_cmd_kd)
        buffer_low_cmd_tau = np.array(buffer_low_cmd_tau)

        buffer_mocap_positions = np.array(buffer_mocap_positions)
        buffer_mocap_orientations = np.array(buffer_mocap_orientations)
        buffer_mocap_linear_velocities = np.array(buffer_mocap_linear_velocities)
        buffer_mocap_angular_velocities = np.array(buffer_mocap_angular_velocities)
        

        import os
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
            
            
        # Save data into npz file
        npz_file_for_this_episode = os.path.join(self.save_path, f"motion_{self.motion_episode_cnt}.npz")
        
        print("\n\n Saving data to ", npz_file_for_this_episode)
        np.savez(
            npz_file_for_this_episode, 
            time=buffer_timestamps, 
            joint_pos=buffer_low_state_motor_positions, 
            joint_vel=buffer_low_state_motor_velocities, 
            tau_est=buffer_low_state_motor_torques,
            IMU_quaternion=buffer_low_state_imu_quaternion, 
            IMU_gyro=buffer_low_state_imu_gyroscope, 
            IMU_acc=buffer_low_state_imu_accelerometer,
            # foot_force=buffer_low_state_foot_force,
            # foot_force_est=buffer_low_state_foot_force_est,
            joint_pos_cmd=buffer_low_cmd_q,
            joint_vel_cmd=buffer_low_cmd_dq,
            kp=buffer_low_cmd_kp,
            kd=buffer_low_cmd_kd,
            tau_cmd=buffer_low_cmd_tau,
            pos=buffer_mocap_positions,
            quat=buffer_mocap_orientations,
            lin_vel=buffer_mocap_linear_velocities,
            ang_vel=buffer_mocap_angular_velocities
        )

        self.get_logger().info(f"Data saved to {self.save_path}.")
        # print first 10 timestamps for all data
        data = np.load(npz_file_for_this_episode, allow_pickle=True)
        print("First 10 timestamps for all data:")
        for key in data.files:
            print(key, data[key][:10])

def main(args=None):

    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    exp_name = args.exp_name

    save_path = './../humanoidverse/logs/delta_a_realdata'  # Default save path, enter folder sim2real to run this
    save_path = os.path.join(save_path, exp_name)
    current_timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_path, current_timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    # Initialize Unitree SDK
    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])
    

    # Initialize rclpy and create the DataLogger node
    rclpy.init(args=None)
    data_logger = DataLogger(save_path=save_path, config=config)

    try:
        rclpy.spin(data_logger)
    except KeyboardInterrupt:
        data_logger.sigINT_handler(None, None)
    finally:
        data_logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()