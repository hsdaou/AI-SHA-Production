from setuptools import find_packages, setup

package_name = 'tts_elevenlabs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'elevenlabs'],
    zip_safe=True,
    maintainer='pi5',
    maintainer_email='pi5@todo.todo',
    description='ROS2 TTS Speaker Node using Eleven Labs API',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tts_elevenlabs_node = tts_elevenlabs.tts_elevenlabs_node:main',
        ],
    },
)
