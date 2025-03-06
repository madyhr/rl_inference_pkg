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
from keyboard_msgs.msg import Key
from motion_stack.core.utils.joint_state import JState
from motion_stack.api.ros2.joint_api import JointHandler, JointSyncerRos
from motion_stack.ros2.utils.conversion import ros_to_time

from rl_inference_pkg.hero_vehicle_policy import HeroVehiclePolicy

# Operator specific namespace for keyboard control
operator = str(os.environ.get("OPERATOR"))
INPUT_NAMESPACE = f"/{operator}"

class RlPolicyNode(Node):
    '''
    Subscribes to the topics containing observation data and sends corresponding actions through Motion Stack API.
    Uses the pre-trained policy given by self.policy to convert these observations to actions. 
    '''

    def __init__(self, policy_path: str, limbs: List[int]):
        super().__init__("rl_policy")

        ##
        # Subscribers
        ##

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

        self.key_press_subscriber_ = self.create_subscription(
            Key, 
            f"{INPUT_NAMESPACE}/keydown", 
            self.key_press_listener_callback, 
            10
        )

        ##
        # Policy specific
        ##

        self.policy = HeroVehiclePolicy(policy_path=policy_path)
        self.base_velocity: List[float] = [0.0] * 6
        self.command: List[float] = [0.0] * 3

        ## 
        # Motion Stack joint interface
        ##
        
        self.front_wheel: JointHandler = JointHandler(self, limbs[0])
        self.rear_wheel: JointHandler = JointHandler(self, limbs[1])
        self.leg: JointHandler = JointHandler(self, limbs[2])
        # joint syncer to use safe trajectories
        self.leg_sync: JointSyncerRos = JointSyncerRos([self.leg])

        ##
        # Params
        ##

        # max cmd vel in m/s and rad/s, based on RL training and hardware
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

        ##
        # Flags
        ##

        self.inference_flag: bool = False
        self.limb_futures = self.limb_setup()
        self.limbs_ready: bool = False
        ##
        # Timer
        ##

        self.timer = self.create_timer(0.02,self.timer_callback) # 50 Hz
        
    def limb_setup(self):
        
        self.get_logger().info(f"Trying to setup the following joints: {self.JOINT_ORDER}")

        front_wheel_setup = self.front_wheel.ready_up(set(self.JOINT_ORDER[0:2]))
        rear_wheel_setup = self.rear_wheel.ready_up(set(self.JOINT_ORDER[2:4]))
        leg_setup = self.leg.ready_up(set(self.JOINT_ORDER[4:9]))

        return [front_wheel_setup, rear_wheel_setup, leg_setup]
    
    def check_limb_ready(self):
        
        # shortcut
        if self.limbs_ready:
            return
        
        # all limbs should be ready 
        if sum(future[0].done() for future in self.limb_futures) != len(self.limb_futures):
            return
        
        self.get_logger().info(f"All limbs are ready.")
        self.limbs_ready = True

        pass

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

    def key_press_listener_callback(self, msg: Key):
        key_code = msg.code
        key_modifier = msg.modifiers

        if key_code == Key.KEY_R:
            self.get_logger().info("'R' received: Inference has BEGUN.")
            self.inference_flag = True

        if key_code == Key.KEY_S:
            self.get_logger().info("'S' received: Inference has STOPPED.")
            self.stop_wheels()
            self.inference_flag = False

    def stop_wheels(self):
        # send a zero velocity command to all 4 wheels
        ros_now = ros_to_time(self.get_clock().now())
        front_wheel_cmd = {self.JOINT_ORDER[i]: 0.0 for i in range(0, 2)}
        rear_wheel_cmd = {self.JOINT_ORDER[i]: 0.0 for i in range(2, 4)}

        self.front_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in front_wheel_cmd.items()])
        self.rear_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in rear_wheel_cmd.items()])

    def send_action(self, action: List[float]):
        
        ros_now = ros_to_time(self.get_clock().now())
        front_wheel_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(0, 2)}
        rear_wheel_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(2, 4)}
        leg_cmd = {self.JOINT_ORDER[i]: action[i] for i in range(4, 9)}

        # add offset back
        leg_cmd["leg1joint4"] += 1.57075

        # send the commands using the dictionary
        self.front_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in front_wheel_cmd.items()])
        self.rear_wheel.send([JState(name=name, time=ros_now, velocity=value) for name, value in rear_wheel_cmd.items()])
        # keep joints outside action space still
        leg_no_cmd = {js.name: js.position for js in self.leg.states if js in self.JOINT_STILL}
        
        self.leg_sync.lerp(leg_cmd | leg_no_cmd)

        # self.get_logger().info("Action successfully sent.")

    def get_states(self) -> Dict[str, JState]:
        out = {}
        out.update({v.name:v for v in self.front_wheel.states})
        out.update({v.name:v for v in self.rear_wheel.states})
        out.update({v.name:v for v in self.leg.states})
        return copy.deepcopy(out)

    def timer_callback(self):
        
        self.check_limb_ready()
        
        if not self.limbs_ready:
            return

        if not self.inference_flag:
            return

        # self.get_logger().info(f"Command: {self.command}")
        # self.get_logger().info(f"Base velocity: {self.base_velocity}")

        states = self.get_states()
        
        useful_states = [states.get(k) for k in self.JOINT_ORDER]

        if None in useful_states:
            self.get_logger().info(f"None in useful states")
            return
        
        # velocity may be None if no velocity command has been given 
        joint_pos = [v.position for v in useful_states]
        joint_vel = [v.velocity if v.velocity is not None else 0.0 for v in useful_states]

        obs = self.policy.compose_observation(
            base_velocity=np.array(self.base_velocity),
            joint_positions=np.array(joint_pos),
            joint_velocities=np.array(joint_vel),
            command = np.array(self.command),
        )

        action = self.policy.get_action(obs)

        self.get_logger().info(f"Observation being received: {obs[15:24]}")
        self.get_logger().info(f"Action being sent: {action}")

        self.send_action(action)

        self.leg_sync.execute()

def main(args=None):
    rclpy.init(args=args)

    # pretrained RL policy
    policy_path = "/home/madyhr/Motion-Stack/src/rl_inference_pkg/policy/policy.pt"
    
    # limb numbers/id in order (front wheel, back wheel, bridge leg)
    limbs = [11, 12, 1]
    node = RlPolicyNode(policy_path, limbs)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()