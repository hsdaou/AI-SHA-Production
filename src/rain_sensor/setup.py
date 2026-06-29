from setuptools import setup
import os
from glob import glob

package_name = 'rain_sensor'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi5',
    maintainer_email='pi5@todo.todo',
    description='ROS 2 driver for rain sensor via Arduino (MH-sensor-series)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rain_sensor_node = rain_sensor.rain_sensor_node:main',
        ],
    },
)
