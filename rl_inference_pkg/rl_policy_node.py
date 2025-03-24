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

from rl_inference_pkg.policies import HeroVehiclePolicy
from rl_inference_pkg.utils import dict_to_array, wrap_to_pi

# Operator specific namespace for ROS2 keyboard control
operator = str(os.environ.get("OPERATOR"))
INPUT_NAMESPACE = f"/{operator}"

class RlPolicyNode(Node):
    '''
    Subscribes to the topics containing observation data and sends corresponding actions through Motion Stack API.
    Uses the pre-trained policy given by self.policy to convert these observations to actions. 
    '''

    def __init__(self, policy_name: str, limbs: List[int]):
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

        # *ordered* joint names for vehicle mode policy

        self.FRONT_WHEEL_JOINTS: List[str] = [
            f"wheel{limbs[0]}_left_joint",
            f"wheel{limbs[0]}_right_joint",
        ]

        self.REAR_WHEEL_JOINTS: List[str] = [
            f"wheel{limbs[1]}_left_joint",
            f"wheel{limbs[1]}_right_joint",
        ]

        self.LEG_JOINTS: List[str] = [
            f"leg{limbs[2]}joint1",
            f"leg{limbs[2]}joint7",
        ]

        self.JOINT_ORDER: List[str] = self.FRONT_WHEEL_JOINTS + self.REAR_WHEEL_JOINTS + self.LEG_JOINTS

        # dictionary of offsets for joints between robot used for training and real robot
        self.JOINT_OFFSET_DICT: Dict[str, float] = dict(zip(self.JOINT_ORDER, [0.0]*len(self.JOINT_ORDER)))

        # specify offsets here
        # self.JOINT_OFFSET_DICT[f"leg{limbs[2]}joint4"] = 1.57078

        self.JOINT_OFFSET = dict_to_array(self.JOINT_OFFSET_DICT,self.JOINT_ORDER)

        # joints to keep still (i.e. they are outside action space), order does not matter here
        self.JOINT_STILL = [
            
            f"leg{limbs[2]}joint2",
            f"leg{limbs[2]}joint6",
            f"leg{limbs[2]}joint4",
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
        # Policy specific
        ##

        self.policy = HeroVehiclePolicy(
            policy_name=policy_name, 
            joint_offset = self.JOINT_OFFSET
        )

        self.base_velocity: List[float] = [0.0] * 6
        self.command: List[float] = [0.0] * 3

        self.policy.prepare_observation_dims(
            n_pos=len(self.LEG_JOINTS),
            n_vel=len(self.FRONT_WHEEL_JOINTS + self.REAR_WHEEL_JOINTS),
            cmd_dim=len(self.command),
            base_vel_dim=len(self.base_velocity)
        )

        # max cmd vel in m/s and rad/s, based on RL training and hardware
        self.CMD_LIN_VEL_MAX = self.policy.CMD_LIN_VEL_MAX
        self.CMD_ANG_VEL_MAX = self.policy.CMD_ANG_VEL_MAX

        ##
        # Timer
        ##

        self.timer = self.create_timer(0.02, self.timer_callback) # 50 Hz
        
    def limb_setup(self):
        """Initiates the limb setup procedure in Motion Stack API and returns list of Future's for the limbs."""
        self.get_logger().info(f"Trying to setup the following joints: {self.JOINT_ORDER}")

        front_wheel_setup = self.front_wheel.ready_up(set(self.FRONT_WHEEL_JOINTS))
        rear_wheel_setup = self.rear_wheel.ready_up(set(self.REAR_WHEEL_JOINTS))
        leg_setup = self.leg.ready_up(set(self.LEG_JOINTS))

        return [front_wheel_setup, 
                rear_wheel_setup, 
                leg_setup
                ]
    
    def check_limb_ready(self):
        """Checks if all limbs on the robot are ready. 
        If yes, then sets the limbs_ready flag to True and allows for policy inference."""
        # shortcut
        if self.limbs_ready:
            return
        
        # ALL limbs need to be ready
        if sum(future[0].done() for future in self.limb_futures) != len(self.limb_futures):
            return
        
        # self.get_logger().info(f"All limbs are ready.")
        self.limbs_ready = True
        self.get_logger().info(f"Test: Setup successful. RL Inference is ready.")
        self.get_logger().info(f"Begin inference by pressing 'B' and halt by pressing 'H'.")
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

        if key_code == Key.KEY_B:
            self.get_logger().info("'B' received: Inference has BEGUN.")
            self.inference_flag = True

        if key_code == Key.KEY_H:
            self.get_logger().info("'H' received: Inference has HALTED.")
            self.stop_wheels()
            self.inference_flag = False

    def send_jstate(self, joint_handler: JointHandler, action_dict: Dict[str, float], action_type: str):
        """ Sends a JState to a JointHandler using the given dictionary of actions. UNSAFE."""
        ros_now = ros_to_time(self.get_clock().now())
        
        # send correct JState depending on action type (action_type)
        if action_type == "velocity":
            joint_handler.send([JState(name=name, time=ros_now, velocity=value) for name, value in action_dict.items()])
        elif action_type == "position":
            # THIS CAN BE VERY UNSAFE. DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING. USE 'JointSyncerRos()' INSTEAD.
            joint_handler.send([JState(name=name, time=ros_now, position=value) for name, value in action_dict.items()])
        else:
            print("Action type was not recognized. Use either 'velocity' or 'position'. ")
            return 

    def stop_wheels(self):
        # send a zero velocity command to all 4 wheels
        front_wheel_action = {joint: 0.0 for joint in self.FRONT_WHEEL_JOINTS}
        rear_wheel_action = {joint: 0.0 for joint in self.REAR_WHEEL_JOINTS}

        self.send_jstate(self.front_wheel, front_wheel_action, action_type="velocity")
        self.send_jstate(self.rear_wheel, rear_wheel_action, action_type="velocity")

    def send_action(self, action: List[float]):
        
        joint_actions = {j: a for (j,a) in zip(self.JOINT_ORDER, action)}

        # add offsets from training robot to real robot
        for joint in self.JOINT_ORDER:
            joint_actions[joint] += self.JOINT_OFFSET_DICT[joint]

        front_wheel_action = {joint: joint_actions[joint] for joint in self.FRONT_WHEEL_JOINTS}
        rear_wheel_action = {joint: joint_actions[joint] for joint in self.REAR_WHEEL_JOINTS}
        leg_action = {joint: joint_actions[joint] for joint in self.LEG_JOINTS}
        # send the commands using the dictionaries
        self.send_jstate(self.front_wheel, front_wheel_action, action_type="velocity")
        self.send_jstate(self.rear_wheel, rear_wheel_action, action_type="velocity")

        # keep joints outside action space still
        leg_no_action = {js.name: js.position for js in self.leg.states if js in self.JOINT_STILL}
        # self.send_jstate(self.leg, leg_action, action_type="position")
        # self.send_jstate(self.leg, leg_no_action, action_type="position")
        self.leg_sync.lerp(leg_action | leg_no_action)
        # TODO: check for: SensorSyncWarning: Syncer is out of sync with sensor data. Call `syncer.clear()` to reset the syncer onto the sensor position.
        # self.get_logger().info("Action successfully sent.")
        # self.get_logger().info(f"Action being sent: {leg_action}")

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
            self.get_logger().info(f"'None' in useful states")
            return
        
        # velocity elements may be None if no velocity command has been given 
        joint_pos = [v.position for v in useful_states]
        joint_vel = [v.velocity if v.velocity is not None else 0.0 for v in useful_states]

        obs = self.policy.compose_observation(
            joint_positions=np.array(wrap_to_pi(joint_pos)),
            joint_velocities=np.array(joint_vel),
            command = np.array(self.command),
            # base_velocity=np.array(self.base_velocity),
        )

        action = self.policy.get_action(obs)

        # self.get_logger().info(f"Observation being received: {obs[:12]}")
        # self.get_logger().info(f"Action being sent: {action}")

        self.send_action(action)
        # step the Motion Stack JointSyncer
        self.leg_sync.execute()

def main(args=None):
    rclpy.init(args=args)

    # pretrained RL policy found in pkg/policy/ directory
    policy_name = "20250319_17_policy.pt"
    
    # limb numbers/id in order (front wheel, back wheel, bridge leg)
    limbs = [12, 14, 1]
    node = RlPolicyNode(policy_name, limbs)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        # if the node encounters error, stop wheels before suicide
        node.stop_wheels()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()