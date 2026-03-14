# Use the ROS 2 Humble base image (works well on ARM64/Raspberry Pi)
FROM ros:humble-ros-base

# Set non-interactive frontend to avoid prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install RealSense camera node and image transport plugins
RUN apt-get update && apt-get install -y \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-image-transport-plugins \
    && rm -rf /var/lib/apt/lists/*

# Source the ROS 2 setup script automatically when the container starts
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Default command (overridden by docker-compose)
CMD ["bash"]
