from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'stt_node'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin-robot',
    maintainer_email='orin-robot@todo.todo',
    description='NVIDIA Canary-1B-v2 STT node for ReSpeaker mic array',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stt_node = stt_node.stt_node:main',
            'stt_assemblyai = stt_node.stt_assemblyai:main',
            'stt_node_api = stt_node.stt_node_api:main',
        ],
    },
)
