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



class RosPolicy(RLPolicy):
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
    
        # Subscribers for /cmd_vel and reset_robot_state
        self.cmd_vel_sub = self.node.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 1
        )
        self.reset_sub = self.node.create_subscription(
            Bool, "reset_robot", self.reset_callback, 1
        )
        self.cmd_vel_last_received_time = None
        self.check_vel_timer = self.node.create_timer(0.1, self.check_cmd_vel_timeout)

    def cmd_vel_callback(self, msg):
        self.use_policy_action = True
        self.get_ready_state = False
        self.cmd_vel_last_received_time = self.node.get_clock().now().nanoseconds / 1e9
        self.lin_vel_command[0, 0] = msg.linear.x
        self.lin_vel_command[0, 1] = msg.linear.y
        self.ang_vel_command[0, 0] = msg.angular.z
        # print(f"Linear velocity command: {self.lin_vel_command}")

    def reset_callback(self, msg):
        if msg.data:
            self.get_ready_state = True
            self.use_policy_action = False
            self.init_count = 0
            self.node.get_logger().info("Setting to init state")
        # print("rest_callback, msg.data: ", msg.data)

    def check_cmd_vel_timeout(self):
        current_time = self.node.get_clock().now().nanoseconds / 1e9
        if self.cmd_vel_last_received_time is not None and (current_time - self.cmd_vel_last_received_time) > 1.0:
            # if timeout occurs, stop and reset the robot
            self.use_policy_action = False
            self.get_ready_state = True

        if self.cmd_vel_last_received_time is None:
            # if no cmd_vel received, do not use policy action
            self.use_policy_action = False

        


# a ros version of sending keyboard commands, need to run RLpolicyRosCmd at the same time
class KeyboardRosPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.reset_publisher_ = self.create_publisher(Bool, 'reset_robot', 10)
        self.lin_vel_command = [0.0, 0.0]  # [vx, vy]
        self.ang_vel_command = 0.0  # [wz]
        self.publish_enabled = False
        
        # Start key listener in a separate thread
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        print("Key listener started")

    def start_key_listener(self):
        """Start a key listener using pynput."""

        def on_press(key):
            try:
                
                print(f"Key {key.char} pressed")

                if key.char == "w":
                    self.lin_vel_command[0] += 0.1
                elif key.char == "s":
                    self.lin_vel_command[0] -= 0.1
                elif key.char == "a":
                    self.lin_vel_command[1] += 0.1
                elif key.char == "d":
                    self.lin_vel_command[1] -= 0.1
                elif key.char == "q":
                    self.ang_vel_command -= 0.1
                elif key.char == "e":
                    self.ang_vel_command += 0.1
                elif key.char == "z":
                    self.ang_vel_command = 0.0
                    self.lin_vel_command = [0.0, 0.0]
                elif key.char == "p":
                    self.publish_enabled = True
                    self.get_logger().info("Starting cmd_vel publishing")
                elif key.char == "o":
                    self.publish_enabled = False
                    self.get_logger().info("Stopping cmd_vel publishing")
                elif key.char == "i":
                    reset_msg = Bool()
                    self.publish_enabled = False
                    reset_msg.data = True
                    self.reset_publisher_.publish(reset_msg)
                    self.get_logger().info("Reset robot state command sent")
                
                if self.publish_enabled:
                    self.publish_cmd_vel()
                print(f"Linear velocity command: {self.lin_vel_command}")
                print(f"Angular velocity command: {self.ang_vel_command}")

            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

    def publish_cmd_vel(self):
        msg = Twist()
        msg.linear.x = self.lin_vel_command[0]
        msg.linear.y = self.lin_vel_command[1]
        msg.angular.z = self.ang_vel_command
        self.publisher_.publish(msg)
    
    def main_loop(self):
        while True:
            if self.publish_enabled:
                self.publish_cmd_vel()
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    keyboard_publisher = KeyboardRosPublisher()
    try:
        keyboard_publisher.main_loop()
    except KeyboardInterrupt:
        pass
    keyboard_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()