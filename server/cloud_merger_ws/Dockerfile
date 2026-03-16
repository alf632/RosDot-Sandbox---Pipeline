# Use ROS 2 Humble base
FROM ros:humble-ros-base

# Install PCL, message filters, and TF2 dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-tf2-sensor-msgs \
    libpcl-dev \
    ros-humble-depth-image-proc \
    ros-humble-compressed-depth-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Set up the colcon workspace
WORKDIR /ros2_ws
COPY ./src /ros2_ws/src

# Build the workspace
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Source the workspace on container start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Run the node by default
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 run cloud_merger merge_node"]
