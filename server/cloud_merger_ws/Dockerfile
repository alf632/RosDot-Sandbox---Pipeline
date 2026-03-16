# Use ROS 2 Jazzy base
FROM ros:jazzy-ros-base

# Install PCL, message filters, and TF2 dependencies
RUN apt-get update && apt-get install -y \
    ros-jazzy-pcl-conversions \
    ros-jazzy-pcl-ros \
    ros-jazzy-message-filters \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-sensor-msgs \
    libpcl-dev \
    libopencv-dev \
    ros-jazzy-depth-image-proc \
    ros-jazzy-compressed-depth-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Set up the colcon workspace
WORKDIR /ros2_ws
COPY ./src /ros2_ws/src

# Build the workspace
RUN bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Source the workspace on container start
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Run the node by default
CMD ["bash", "-c", "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && ros2 run cloud_merger merge_node"]
