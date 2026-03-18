# Use ROS 2 Jazzy Base
FROM ros:jazzy-ros-base

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    libusb-1.0-0 \
    ros-jazzy-composition-interfaces \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-geometry \
    ros-jazzy-message-filters \
    ros-jazzy-apriltag-ros \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install pyrealsense2 via pip
# --break-system-packages is required on Ubuntu 24.04/Jazzy when not using venv
RUN pip3 install pyrealsense2 --break-system-packages

# Setup workspace
WORKDIR /ros2_ws
COPY . /ros2_ws/src/sandbox_operator

# Create the required empty resource file for ament_python
RUN mkdir -p /ros2_ws/src/sandbox_operator/resource && \
    touch /ros2_ws/src/sandbox_operator/resource/sandbox_operator

# Build the workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

# Setup entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command (Run Operator)
CMD ["ros2", "run", "sandbox_operator", "operator"]
