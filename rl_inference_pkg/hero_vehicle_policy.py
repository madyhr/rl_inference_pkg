from typing import List

import numpy as np
import torch
import quaternion

import os
import ament_index_python.packages
class HeroVehiclePolicy(object):
    """The Hero Vehicle running a Locomotion Policy"""

    def __init__(
        self,
        policy_file = None,
    ) -> None:
        policy_path = self.load_policy_path()
        self.policy = torch.jit.load(policy_path)
        self._pos_action_scale = 0.5
        self._vel_action_scale = 5.0
        self._previous_action = np.zeros(9)

    def load_policy_path(self):
        """
        Loads the path to the policy.pt file within the 'my_package' package.

        Returns:
            str: Absolute path to the policy.pt file, or None if not found.
        """
        try:
            package_path = ament_index_python.packages.get_package_share_directory('rl_inference_pkg')
            policy_path = os.path.join(package_path, 'policy', 'policy.pt')

            if os.path.exists(policy_path):
                return policy_path
            else:
                print(f"Error: policy.pt not found at {policy_path}")
                return None

        except ament_index_python.packages.PackageNotFoundError:
            print("Error: pkg not found.")
            return None

    def compose_observation(self, 
                            base_velocity: np.ndarray, 
                            joint_positions: np.ndarray, 
                            joint_velocities: np.ndarray, 
                            command: np.ndarray) -> np.ndarray:
        """
        Compose the observation vector for the policy.
        
        Note that joint order should be:
        wheel11_left_joint
        wheel11_right_joint
        wheel12_left_joint
        wheel12_right_joint
        leg1joint1
        leg1joint2
        leg1joint4
        leg1joint6
        leg1joint7

        Argument:
        base_velocity (np.ndarray, dim = 6) -- the linear and angular velocity of the base link.
        joint_positions (np.ndarray, dim = 9) -- the joint positions in radians
        joint_velocities (np.ndarray, dim = 9) -- the joint velocities in radians/second
        command (np.ndarray, dim = 3) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel = base_velocity[:3]
        ang_vel = base_velocity[3:6]

        obs = np.zeros(36)
        # Base lin vel
        obs[:3] = lin_vel
        # Base ang vel
        obs[3:6] = ang_vel
        # Joint states
        obs[6:15] = joint_positions
        obs[15:24] = joint_velocities
        # Command
        obs[24:27] = command
        # Previous Action
        obs[27:36] = self._previous_action
        
        return obs

    def get_action(self, obs: np.ndarray) -> List[float]:
        """
        Get action according to observation using model inference. Scale action according to action scales from learning env.

        Argument:
        obs (np.ndarray, dim = 36) -- Observations composed in the manner described in compose_observation().

        Returns:
        action (np.ndarray, dim = 9) -- Action taken 
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        self._previous_action = action

        scaled_action = [0] * len(action)
        scaled_action[:4] = action[:4] * self._vel_action_scale
        scaled_action[4:9] = action[4:9] * self._pos_action_scale

        # apply offsets for real robot
        scaled_action[6] = scaled_action[6] + 1.57075

        return list(scaled_action)

