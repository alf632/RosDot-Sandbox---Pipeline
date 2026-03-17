from setuptools import setup
import os
from glob import glob

package_name = 'sandbox_operator'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f'{package_name}.loaders'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyrealsense2', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='Sandbox Dev',
    description='Unified Operator and Calibration TUI for AR Sandbox',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'operator = sandbox_operator.unified_operator:main',
            'calibrate = sandbox_operator.calibration_tui:main'
        ],
    },
)