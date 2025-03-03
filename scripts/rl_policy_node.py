#!/usr/bin/env python3

from typing import List, Dict
import copy
import numpy as np

import rclpy 
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist, TFStamped
from motion_stack.core.utils.joint_state import JState
from motion_stack.api.ros2.joint_api import JointHandler
from motion_stack.ros2.utils.conversion import ros_to_time

from hero_vehicle_policy import HeroVehiclePolicy

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
            10)
        
        self.base_vel_subscriber_  = self.create_subscription(
            TFStamped, 
            f'/{base_name}_base_vel', 
            self.base_vel_listener_callback, 
            10)

        self.policy = policy
        self.front_wheel = JointHandler(self, limbs[0])
        self.rear_wheel = JointHandler(self, limbs[1])
        self.leg = JointHandler(self, limbs[2])

        self.action = None
        self.base_velocity= None
        self.command = None

        # max cmd vel in m/s and rad/s
        self.CMD_LIN_VEL_MAX = 0.12
        self.CMD_ANG_VEL_MAX = 0.25

        # ordered joint names for vehicle mode policy
        self.JOINT_ORDER = [
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

    def cmd_vel_listener_callback(self, msg):
        # vel command only has 3 dims, x, y and rz. 
        self.command = [np.clip(msg.linear.x, self.CMD_LIN_VEL_MAX, -self.CMD_LIN_VEL_MAX),
                        np.clip(msg.linear.y, self.CMD_LIN_VEL_MAX, -self.CMD_LIN_VEL_MAX), 
                        np.clip(msg.angular.z, self.CMD_ANG_VEL_MAX, -self.CMD_ANG_VEL_MAX)]

    def base_vel_listener_callback(self, msg):
        # base velocity given as a translation and rotation (implicitly over time, see base_velocity_node.py)
        self.base_velocity = [msg.transform.translation.x,
                              msg.transform.translation.y,
                              msg.transform.translation.z, 
                              msg.transform.rotation.x,
                              msg.transform.rotation.y,
                              msg.transform.rotation.z,
                              msg.transform.rotation.w]

    def send_action(self, action):
        
        # check all joints' Future if they are ready. 
        if not (self.front_wheel.ready.done() and self.rear_wheel.ready.done() and self.leg.ready.done()):
            return
        ##
        # add check if velocity is prrrrrrrrrrrrrrrroperly sent without the position
        ##
        ros_now = ros_to_time(self.get_clock().now())
        front_wheel_cmd = [JState(name=self.JOINT_ORDER[i], time=ros_now, velocity=action[i]) for i in range(0,2)]
        rear_wheel_cmd = [JState(name=self.JOINT_ORDER[i], time=ros_now, velocity=action[i]) for i in range(2,4)]
        leg_cmd = [JState(name=self.JOINT_ORDER[i], time=ros_now, position=action[i]) for i in range(4,9)]

        self.front_wheel.send(front_wheel_cmd)
        self.rear_wheel.send(rear_wheel_cmd)
        self.leg.send(leg_cmd)

        # keep joints outside action space still
        leg_js_dict = {js.name: js.position for js in self.leg.states}
        leg_no_cmd = [JState(name=self.JOINT_STILL[i], time=ros_now, position = leg_js_dict[self.JOINT_STILL[i]]) for i in range(len(self.JOINT_STILL))]
        self.leg.send(leg_no_cmd)

    def get_states_pos(self) -> Dict[str, JState]:
        out = {}
        out.update(self.front_wheel.states)
        out.update(self.rear_wheel.states)
        out.update(self.leg.states)
        return copy.deepcopy(out)
        
    def timer_callback(self):

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