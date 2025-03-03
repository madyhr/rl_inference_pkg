#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist, TFStamped, PoseStamped
from std_msgs.msg import String
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException

from motion_stack.ros2.utils.conversion import ros_to_time

import numpy as np
import quaternion as qt


class base_vel_sub_pub_node(Node):
    '''
    Converts a base pose time series from MoCap published at topic '/XXX' to
    a base velocity (Twist) published at topic '/base_vel'
    '''

    def __init__(self,base_name):
        super().__init__("base_vel_sub_pub")

        self.base_pose_subscriber_  = self.create_subscription(
            PoseStamped, 
            f'/mocap_{base_name}', 
            self.listener_callback, 
            10)
        
        self.base_vel_publisher_ = self.create_publisher(
            String,
            f'/{base_name}_base_vel',
            10
        )
        self.data = None
        self.get_logger().info("Waiting for first pose message...")

        while self.data is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.previous_tf = self.data
        self.get_logger().info("Received first pose message, initialization complete.")
        
        self.timer = self.create_timer(0.02,self.timer_callback) # every 20 ms

    def listener_callback(self, msg):
        self.data = msg

    def publish_base_vel(self):
        now = self.get_clock().now()
        msg = TFStamped()

        msg.header.time = now
        msg.transform.translation = self.base_lin_vel
        msg.transform.rotation = self.base_ang_vel

        self.base_vel_publisher_.publish(msg)

    def angular_velocity_quat(q1, q2, dt):
        """
        Computes angular velocity quaternion given previous quat, q1, current quat, q2, and time difference, dt.
        Assumes q1,q2 are given in the format described by 'geometry_msgs/msg/Quaternion orientation' i.e. (x,y,z,w).
        """ 
        return q2 * q1**(-1) / dt

    def timer_callback(self):

        current_tf = self.data
        dt = current_tf.header.time - self.previous_tf.header.time

        self.base_lin_vel = (current_tf.transform.translation - self.previous_tf.transform.translation) / dt

        self.base_ang_vel = self.angular_velocity_quat(
            q1 = self.previous_tf.transform.rotation,
            q2 = current_tf.transform.rotation, 
            dt = dt
        )

        self.publish_base_vel()

        self.previous_tf = current_tf



def main(args=None):
    rclpy.init(args=args)

    node = base_vel_sub_pub_node(base_name="leg1link4")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()