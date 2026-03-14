# Use the ROS 2 Humble base image
FROM ros:humble-ros-base

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install GridMap messages, OpenCV, and Numpy
RUN apt-get update && apt-get install -y \
    ros-humble-grid-map-msgs \
    python3-numpy \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for our app and copy the script in
WORKDIR /app
COPY heightmap_udp_streamer.py /app/heightmap_udp_streamer.py

# Make sure the script is executable
RUN chmod +x /app/heightmap_udp_streamer.py

# Source ROS 2 and run the script
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 /app/heightmap_udp_streamer.py"]
