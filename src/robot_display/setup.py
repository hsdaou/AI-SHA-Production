import os
from glob import glob
from setuptools import setup

package_name = 'robot_display'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'local_setup.dsv']),
        (os.path.join('share', package_name, 'hook'),
         ['hooks/ament_prefix_path.dsv', 'hooks/ament_prefix_path.sh']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi4',
    maintainer_email='pi4@localhost',
    description='Robot Speech Display - Shows STT input and LLM responses',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'display_node = robot_display.display_node:main',
        ],
    },
)
