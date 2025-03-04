from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rl_inference_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'policy'), glob(os.path.join('policy', '*.pt')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marcus',
    maintainer_email='marcusdyhr@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f"mocap_base_vel_node = {package_name}.mocap_base_vel_node:main",
            f"rl_policy_node = {package_name}.rl_policy_node:main",

        ],
    },
)
