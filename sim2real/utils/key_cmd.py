import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from pynput import keyboard
import threading
import time

import sys
sys.path.append('./rl_inference')

from rl_policy import RLPolicy


class KeyboardPolicy(RLPolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        super().__init__(config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        
        # Start key listener in a separate thread
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()


    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(key):
            try:
                if key.char == "]":
                    self.use_policy_action = True
                    self.get_ready_state = False
                    self.node.get_logger().info("Using policy actions")
                elif key.char == "o":
                    self.use_policy_action = False
                    self.get_ready_state = False
                    self.node.get_logger().info("Actions set to zero")

                elif key.char == "i":
                    self.get_ready_state = True
                    self.init_count = 0
                    self.node.get_logger().info("Setting to init state")
                elif key.char == "w":
                    # print("w pressed")
                    # import ipdb; ipdb.set_trace()
                    self.lin_vel_command[0, 0]+=0.1
                    # print(f"Linear velocity command -----: {self.lin_vel_command}")
                elif key.char == "s":
                    self.lin_vel_command[0, 0]-=0.1
                elif key.char == "a":
                    self.lin_vel_command[0, 1]+=0.1 
                elif key.char == "d":
                    self.lin_vel_command[0, 1]-=0.1
                elif key.char == "q":
                    self.ang_vel_command[0, 0]-=0.1
                elif key.char == "e":
                    self.ang_vel_command[0, 0]+=0.1
                elif key.char == "z":
                    self.ang_vel_command[0, 0] = 0.
                    self.lin_vel_command[0, 0] = 0.
                    self.lin_vel_command[0, 1] = 0.
                elif key.char == "u":
                    self.base_height_command += 0.05
                elif key.char == "l":
                    self.base_height_command -= 0.05
                elif key.char == "[":
                    self.stand_command = 1 - self.stand_command
                    if self.stand_command == 0:
                        self.ang_vel_command[0, 0] = 0.
                        self.lin_vel_command[0, 0] = 0.
                        self.lin_vel_command[0, 1] = 0.
                print(f"Linear velocity command: {self.lin_vel_command}")
                print(f"Angular velocity command: {self.ang_vel_command}")
                print(f"Base height command: {self.base_height_command}")
                print(f"Stand command: {self.stand_command}")
            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive