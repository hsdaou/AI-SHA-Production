from setuptools import find_packages, setup

package_name = 'motor_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi4',
    maintainer_email='pi4@todo.todo',
    description='4-motor control node for HC-160A S2 drivers with encoder feedback',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_node = motor_control.motor_node:main',
            'encoder_node = motor_control.encoder_node:main',
        ],
    },
)
