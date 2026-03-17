FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ros-jazzy-composition-interfaces \
    ros-jazzy-realsense2-camera \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY ./ /ros2_ws/src/sandbox_components

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
