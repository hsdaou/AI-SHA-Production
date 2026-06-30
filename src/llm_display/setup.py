from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'llm_display'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi5',
    maintainer_email='pi5@todo.todo',
    description='LLM Display Node with animated GUI for Raspberry Pi 5',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_display = llm_display.dashboard_display:main',
            'llm_display_chat = llm_display.display_node:main',
            'llm_display_face = llm_display.robot_face_display:main',
            'robot_display_right_eye = llm_display.robot_display_right_eye:main',
        ],
    },
)
