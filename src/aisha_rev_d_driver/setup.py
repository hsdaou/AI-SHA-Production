from glob import glob
import os

from setuptools import find_packages, setup


package_name = "aisha_rev_d_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AI-SHA project",
    maintainer_email="maintainer@example.com",
    description="Fail-safe Rev D differential encoder and odometry adapter",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rev_d_encoder_adapter = aisha_rev_d_driver.node:main",
        ],
    },
)
