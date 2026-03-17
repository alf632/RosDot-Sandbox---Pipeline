#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

CURRENT_HOST=$(hostname)

# Sanitize the hostname (replace '-' with '_') and set the global ROS namespace
CLEAN_NS="/${CURRENT_HOST//-/_}"

echo "========================================="
echo " Container booted in NS: $CLEAN_NS"
echo "========================================="

exec ros2 run rclcpp_components component_container_mt --ros-args -r __ns:=$CLEAN_NS --log-level info
