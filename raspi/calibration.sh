docker run -it --rm \
  --name realsense_cam1_calib \
  --network host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -e ROS_DOMAIN_ID=42 \
  raspi-realsense_streamer \
  bash -c "source /opt/ros/jazzy/setup.bash && \
  ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -r __node:=cam1 \
  -r __ns:=/camera \
  -p camera_name:=cam1 \
  -p initial_reset:=true \
  -p enable_color:=true \
  -p enable_depth:=false \
  -p enable_infra1:=true \
  -p enable_infra2:=false \
  -p depth_module.profile:=848x480x30 \
  -p rgb_camera.profile:=1920x1080x30 \
  -p depth_module.emitter_enabled:=0 \
  -p align_depth.enable:=true"
