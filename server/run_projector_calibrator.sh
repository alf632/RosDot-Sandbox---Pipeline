docker run -it --rm \
  --name projector_calibrator \
  --network host \
  -e ROS_DOMAIN_ID=42 \
  -v $(pwd)/projector_calibrator.py:/tmp/projector_calibrator.py:ro \
  ros:jazzy-ros-base \
  bash -c "apt-get update && \
           apt-get install -y python3-pip ros-jazzy-cv-bridge ros-jazzy-image-geometry ros-jazzy-tf2-ros ros-jazzy-message-filters && \
           pip3 install opencv-contrib-python==4.6.0.66 numpy && \
           source /opt/ros/jazzy/setup.bash && \
           python3 /tmp/projector_calibrator.py"
