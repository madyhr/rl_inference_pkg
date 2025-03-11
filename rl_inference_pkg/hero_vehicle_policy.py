from typing import List, Dict, Optional

import numpy as np
import torch
import quaternion

import os
import ament_index_python.packages
class HeroVehiclePolicy(object):
    """ The Hero Vehicle running a Locomotion Policy
    This policy class expects observations of the following type:
    - joint states: position and velocity of all joints
    - command: 3 dimensional velocity command (v_x, v_y, w_z)
    - base velocity [Optional]: the velocity (v_x, v_y, v_z, w_x, w_y, w_z) of the robot base frame in the world frame 

    Remember to call self.prepare_observation_dims() before calling any of the other functions.
    

    Args:
        policy_name: str -- name of the file containing the policy in the pkg/policy/ directory
        joint_offset: np.ndarray -- array of joint offsets 
    """

    def __init__(
        self,
        policy_name: str = None,
        joint_offset: np.ndarray = None
    ) -> None:
        policy_path = self.load_policy_path(policy_name)
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        self.joint_offset = joint_offset
        self.POS_ACTION_SCALE: float = 0.5
        self.VEL_ACTION_SCALE: float = 5.0
        # self._previous_action: np.ndarray = np.zeros(9)
        self._previous_action: np.ndarray = np.zeros(len(joint_offset))

        self.num_pos_joints: int = 0
        self.num_vel_joints: int = 0
        self.cmd_dim: int = 0
        self.base_vel_dim: int = 0

    def load_policy_path(self, policy_name: str):
        """
        Loads the path to the policy.pt file within the package directory.

        Returns:
            str: Absolute path to the policy.pt file, or None if not found.
        """
        try:
            package_path = ament_index_python.packages.get_package_share_directory('rl_inference_pkg')
            policy_path = os.path.join(package_path, 'policy', f'{policy_name}')

            if os.path.exists(policy_path):
                return policy_path
            else:
                print(f"Error: policy.pt not found at {policy_path}")
                return None

        except ament_index_python.packages.PackageNotFoundError:
            print("Error: pkg not found.")
            return None

    def prepare_observation_dims(self, n_pos, n_vel, cmd_dim, base_vel_dim):
        """Specify dimensions of the different observations for internal use in compose_observation() and get_action().
        
        Args:
            n_vel (int): Number of velocity controlled joints
            n_pos (int): Number of position controlled joints
            cmd_dim (int): Dimension of velocity command (usually 3)
            base_vel_dim (int): Dimension of base velocity observation (usually 6)
        """
        self.num_pos_joints = n_pos
        self.num_vel_joints = n_vel
        self.cmd_dim = cmd_dim
        self.base_vel_dim = base_vel_dim

    def compose_observation(self,  
                            joint_positions: np.ndarray, 
                            joint_velocities: np.ndarray, 
                            command: np.ndarray,
                            base_velocity: Optional[np.ndarray] = None,
                            ) -> np.ndarray:
        """
        Compose the observation array for the policy.

        Args:
            joint_positions (np.ndarray): the joint positions in radians
            joint_velocities (np.ndarray): the joint velocities in radians/second
            command (np.ndarray, dim = 3): the robot command (v_x, v_y, w_z)
            base_velocity (np.ndarray | None): the linear and angular velocity of the base link. 

        Returns:
            obs (np.ndarray): The observation array.

        """
        n_p = self.num_pos_joints
        n_v = self.num_vel_joints
        n = n_p + n_v
        c = self.cmd_dim
        b = self.base_vel_dim

        if base_velocity == None:
            obs = np.zeros(3*n + c)
            # Joint states
            obs[:n] = joint_positions - self.joint_offset
            obs[n:2*n] = joint_velocities
            # Command
            obs[2*n : 2*n+c] = command
            # Previous Action
            obs[2*n+c : 3*n+c] = self._previous_action
        else: 
            obs = np.zeros(3*n + c + b)
            # Joint states
            obs[:b] = base_velocity
            obs[b:b+n] = joint_positions - self.joint_offset
            obs[b+n:b+2*n] = joint_velocities
            # Command
            obs[b+2*n : b+2*n+c] = command
            # Previous Action
            obs[b+2*n+c : b+3*n+c] = self._previous_action
        
        return obs

    def get_action(self, obs: np.ndarray) -> List[float]:
        """
        Get action according to observation using model inference. Scale action according to action scales from learning env.

        Argument:
        obs (np.ndarray) -- Observations composed in the manner described in compose_observation().

        Returns:
        action (np.ndarray) -- Action taken 
        """
        with torch.inference_mode():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()

        self._previous_action = action

        n_p = self.num_pos_joints
        n_v = self.num_vel_joints

        # scale position and velocity actions according to their scales defined during training
        scaled_action = [0.0] * len(action)
        scaled_action[:n_v] = action[:n_v] * self.VEL_ACTION_SCALE
        # scaled_action[n_v:n_v+n_p] = action[n_v:n_v+n_p]  * self.POS_ACTION_SCALE

        return list(scaled_action)

