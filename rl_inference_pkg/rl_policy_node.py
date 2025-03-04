#!/usr/bin/env python3

from typing import List, Dict
import copy
import numpy as np

import os
from ament_index_python.packages import get_package_share_directory

import rclpy 
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist, TwistStamped
from motion_stack.core.utils.joint_state import JState
from motion_stack.api.ros2.joint_api import JointHandler
from motion_stack.ros2.utils.conversion import ros_to_time

from rl_inference_pkg.hero_vehicle_policy import HeroVehiclePolicy

class RlPolicyNode(Node):
    '''
    Subscribes to the topics containing observation data and sends corresponding actions through Motion Stack API.
    Uses the pre-trained policy given by self.policy to convert these observations to actions. 
    '''

    def __init__(self, policy, limbs, base_name):
        super().__init__("rl_policy")

        self.cmd_vel_subscriber_  = self.create_subscription(
            Twist, 
            '/cmd_vel', 
            self.cmd_vel_listener_callback, 
            10
        )
        
        self.base_vel_subscriber_  = self.create_subscription(
            TwistStamped, 
            '/rl_base_vel', 
            self.base_vel_listener_callback, 
            10
        )

        self.get_logger().info("RL Policy Inference has been initialized.")

        self.policy = policy
        self.front_wheel = JointHandler(self, limbs[0])
        self.rear_wheel = JointHandler(self, limbs[1])
        self.leg = JointHandler(self, limbs[2])

        self.base_velocity = None
        self.command = None

        # max cmd vel in m/s and rad/s
        self.CMD_LIN_VEL_MAX = 0.12
        self.CMD_ANG_VEL_MAX = 0.25

        # ordered joint names for vehicle mode policy
        self.JOINT_ORDER: List[str] = [
            f"wheel{limbs[0]}_left_joint",
            f"wheel{limbs[0]}_right_joint",
            f"wheel{limbs[1]}_left_joint",
            f"wheel{limbs[1]}_right_joint",
            f"leg{limbs[2]}joint1",
            f"leg{limbs[2]}joint2",
            f"leg{limbs[2]}joint4",
            f"leg{limbs[2]}joint6",
            f"leg{limbs[2]}joint7",
        ]
        # joints to keep still (i.e. they are outside action space)
        self.JOINT_STILL = [
            f"leg{limbs[2]}joint3",
            f"leg{limbs[2]}joint5",
        ]

        self.timer = self.create_timer(0.02,self.timer_callback) # 200 Hz

    def cmd_vel_listener_callback(self, msg: Twist):
        # vel command only has 3 dims, x, y and rz. 
        self.command = [
            np.clip(msg.linear.x, -self.CMD_LIN_VEL_MAX, self.CMD_LIN_VEL_MAX),
            np.clip(msg.linear.y, -self.CMD_LIN_VEL_MAX, self.CMD_LIN_VEL_MAX), 
            np.clip(msg.angular.z, -self.CMD_ANG_VEL_MAX, self.CMD_ANG_VEL_MAX)
        ]

    def base_vel_listener_callback(self, msg:TwistStamped):
        self.base_velocity = [
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z
        ]

    def send_action(self, action: List[float]):
        
        # check all joints' Future if they are ready. 
        if not (self.front_wheel.ready.done() and self.rear_wheel.ready.done() and self.leg.ready.done()):
            self.get_logger().info("No actions were sent.")
            return
        ##
        # add check if velocity is prrrrrrrrrrrrrrrroperly sent without the position
        ##
        ros_now = ros_to_time(self.get_clock().now())
        front_wheel_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(0, 2)}
        rear_wheel_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(2, 4)}
        leg_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(4, 9)}

        # send the commands using the dictionary
        self.front_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in front_wheel_cmd.items()])
        self.rear_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in rear_wheel_cmd.items()])
        self.leg.send([JState(name=name, time=ros_now, position=value) for name, value in leg_cmd.items()])

        # keep joints outside action space still
        # leg_js_dict = {js.name: js.position for js in self.leg.states.items()}
        leg_js_dict = {js.name: js.position for js in self.leg.states}
        leg_no_cmd = [JState(name=name, time=ros_now, position=leg_js_dict[name]) for name in self.JOINT_STILL]
        self.leg.send(leg_no_cmd)

        self.get_logger().info("Action successfully sent.")

    def get_states_pos(self) -> Dict[str, JState]:
        out = {}
        # out.update({v.name:v for v in self.front_wheel.states.items()})
        # out.update({v.name:v for v in self.rear_wheel.states.items()})
        # out.update({v.name:v for v in self.leg.states.items()})
        out.update({v.name:v for v in self.front_wheel.states})
        out.update({v.name:v for v in self.rear_wheel.states})
        out.update({v.name:v for v in self.leg.states})
        return copy.deepcopy(out)

    def timer_callback(self):

        # self.get_logger().info(f"Command: {self.command}")
        # self.get_logger().info(f"Base velocity: {self.base_velocity}")

        states = self.get_states_pos()

        useful_states = [states.get(k) for k in self.JOINT_ORDER]
        
        if None in useful_states:
            return
        
        joint_pos = [v.position for v in useful_states]
        joint_vel = [v.velocity for v in useful_states]

        obs = self.policy.compose_observation(
            base_velocity=np.array(self.base_velocity),
            joint_positions=np.array(joint_pos),
            joint_velocities=np.array(joint_vel),
            command = np.array(self.command),
        )

        action = self.policy.get_action(obs)

        self.get_logger().info(f"Action being sent: {action}")

        self.send_action(action)

def main(args=None):
    rclpy.init(args=args)

    # pretrained RL policy
    policy = HeroVehiclePolicy(policy_file="policy.pt")
    
    # limb numbers/id in order (front wheel, back wheel, bridge leg)
    limbs = [11, 12, 1]
    base_name = "leg1link4"
    node = RlPolicyNode(policy, limbs, base_name)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()