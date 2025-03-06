from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_name = 'rl_inference_pkg'

    teleop_node = Node(
        package = 'teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name = 'teleop_node',
    )

    base_velocity_node = Node(
        package = pkg_name,
        executable = 'mocap_base_vel_node',
        output = 'screen',
    )

    rl_policy_node = Node(
        package = pkg_name,
        executable = 'rl_policy_node'
    )

    return LaunchDescription([
        # teleop_node,
        # base_velocity_node,
        rl_policy_node
    ])