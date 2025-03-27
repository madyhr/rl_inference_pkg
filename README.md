# RL Inference Package

This package facilitates the inference of a reinforcement learning policy on real hardware using the [Motion Stack by Elian Neppel](https://motion-stack.deditoolbox.fr/). Its main dependencies are the Motion Stack and PyTorch. 


## How it works

The node `rl_policy_node.py` loads a PyTorch JIT policy located in the `policy/` folder. It then uses the joint state feedback from the Motion Stack to forward observations to the policy which in turn outputs actions. These actions are then sent to the robots using `JointHandler` and `JointSyncer` (see the Motion Stack documentation for more info on these). 

> **Note**: Currently only the policies for the Hero Vehicle have been implemented and tested, but you could simply define a new one in `policies.py` and change which "legs" to interface with.

## How to use it

You should clone this package into the same repository in `Motion-Stack/src`. After building and sourcing the workspace, you can then launch the RL inference node with the associated launch script in `launch/`:

```
ros2 launch rl_inference_pkg rl_policy_launch
```

The policies that have been used so far also need a command velocity published as a `Twist` message at `/cmd_vel`. This can easily be done using ``teleop_twist_keyboard` which is run using

```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

However, you can also implement your own method for publishing a command velocity, as it simply listens to the specific topic of `/cmd_vel`. 

The launch file does also include the option to launch a node that converts a time series of MoCap TFs into linear and angular velocities, however it has not been extensively used. This was meant to be a method of measuring base velocity without access to an IMU. You will therefore see remnants of it, like a subscriber to the topic `rl_base_vel`. Future implementations should include an IMU datastream instead. 




