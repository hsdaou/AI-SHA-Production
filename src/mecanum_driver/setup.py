from setuptools import setup
import os
from glob import glob

package_name = 'mecanum_driver'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Mecanum robot motor driver for Arduino over serial',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mecanum_driver = mecanum_driver.mecanum_driver_node:main',
            'motor_test = mecanum_driver.motor_test:main',
            'motor_direct = mecanum_driver.motor_direct:main',
        ],
    },
)
