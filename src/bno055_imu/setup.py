from setuptools import setup
import os
from glob import glob

package_name = 'bno055_imu'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi5',
    maintainer_email='pi5@todo.todo',
    description='ROS 2 driver for BNO055 9-DOF IMU sensor',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bno055_node = bno055_imu.bno055_node:main',
            'imu_bridge = bno055_imu.imu_bridge:main',
            'imu_pose_publisher = bno055_imu.imu_pose_publisher:main',
        ],
    },
)
