#!/usr/bin/env python3
import rclpy 
from rclpy.time import Time
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped, TwistStamped
from std_msgs.msg import String
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException
from motion_stack.ros2.utils.conversion import ros_to_time

import numpy as np
import quaternion


class base_vel_sub_pub_node(Node):
    '''
    Converts a base pose time series from MoCap published at topic '/{tf_name}' to
    a base velocity (Twist) published at topic '/{base_name}_base_vel'
    '''

    def __init__(self,tf_name):
        super().__init__("base_vel_sub_pub")

        self.base_pose_subscriber_ = self.create_subscription(
            PoseStamped, 
            f'/{tf_name}', 
            self.listener_callback, 
            10)
        
        self.base_vel_publisher_ = self.create_publisher(
            TwistStamped,
            f'/base_vel',
            10
        )
        self.data = None
        self.get_logger().info("Waiting for first pose message...")

        while self.data is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.previous_pose: PoseStamped = self.data
        self.get_logger().info("Received first pose message, initialization complete.")
        
        self.timer = self.create_timer(0.02,self.timer_callback) # every 20 ms

    def listener_callback(self, msg):
        self.data: PoseStamped = msg

    def publish_base_vel(self):
        msg = TwistStamped()
        now = self.get_clock().now().to_msg()
        # self.get_logger().info(f"Lin_vel: {self.base_lin_vel}")
        # self.get_logger().info(f"Ang_vel: {self.base_ang_vel}")
        msg.header.stamp = now

        msg.twist.linear.x = self.base_lin_vel[0]
        msg.twist.linear.y = self.base_lin_vel[1]
        msg.twist.linear.z = self.base_lin_vel[2]

        msg.twist.angular.x = self.base_ang_vel[0]
        msg.twist.angular.y = self.base_ang_vel[1]
        msg.twist.angular.z = self.base_ang_vel[2]

        self.base_vel_publisher_.publish(msg)

    def time_difference(self, pose1: PoseStamped, pose2: PoseStamped):
        t1 = Time.from_msg(pose1.header.stamp)
        t2 = Time.from_msg(pose2.header.stamp)
        # time difference in seconds
        dt = (t2 - t1).nanoseconds * 1e-9  
        return dt

    def angular_velocity_quat(self, current_pose: PoseStamped, previous_pose: PoseStamped, dt: float):
        """
        Computes angular velocity quaternion given current pose and time difference, dt.
        Careful: numpy-quaternion uses (w,x,y,z) format while ROS2 uses (x,y,z,w) format. 

        Returns:
        List[float] -- angular velocity
        """ 
        if not dt > 0:
            return [0.0, 0.0, 0.0]

        current_q = current_pose.pose.orientation
        previous_q = previous_pose.pose.orientation

        q1 = np.quaternion(previous_q.w, previous_q.x, previous_q.y, previous_q.z)
        q2 = np.quaternion(current_q.w, current_q.x, current_q.y, current_q.z)
        
        R = np.array([q1, q2])  
        t = np.array([0, dt])  

        omega: np.ndarray = quaternion.angular_velocity(R, t)

        # last entry corresponds to the current angular velocity
        return list(omega[-1])

        return 2 * q2 * q1**(-1) / dt

    def linear_velocity(self, current_pose: PoseStamped, previous_pose:PoseStamped, dt: float):
        
        if not dt > 0:  
            return [0.0, 0.0, 0.0]  
        
        lin_vel = [
            (current_pose.pose.position.x - previous_pose.pose.position.x) / dt,
            (current_pose.pose.position.y - previous_pose.pose.position.y) / dt,
            (current_pose.pose.position.z - previous_pose.pose.position.z) / dt
        ]

        return lin_vel            

    def timer_callback(self):

        current_pose = self.data
        dt = self.time_difference(self.previous_pose, current_pose)

        self.base_lin_vel = self.linear_velocity(current_pose, self.previous_pose, dt)
        self.base_ang_vel = self.angular_velocity_quat(current_pose, self.previous_pose, dt)

        self.publish_base_vel()

        self.previous_pose = current_pose



def main(args=None):
    rclpy.init(args=args)

    node = base_vel_sub_pub_node(tf_name="mocap4gripper2_straight/pose")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()