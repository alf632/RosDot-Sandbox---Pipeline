docker run -it --rm \
  --name projector_calibrator \
  --network host \
  -e ROS_DOMAIN_ID=42 \
  -v $(pwd)/projector_calibrator.py:/tmp/projector_calibrator.py:ro \
  ros:humble-ros-base \
  bash -c "apt-get update && \
           apt-get install -y python3-pip ros-humble-cv-bridge ros-humble-image-geometry ros-humble-tf2-ros ros-humble-message-filters && \
           pip3 install opencv-contrib-python==4.6.0.66 numpy && \
           source /opt/ros/humble/setup.bash && \
           python3 /tmp/projector_calibrator.py"
