docker run -it --rm \
  --name realsense_cam1_calib \
  --network host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -e ROS_DOMAIN_ID=42 \
  <your_image_name> \
  bash -c "source /opt/ros/humble/setup.bash && \
  ros2 launch realsense2_camera rs_launch.py \
  camera_name:=cam1 \
  enable_color:=true \
  enable_depth:=false \
  enable_infra1:=true \
  enable_infra2:=false \
  rgb_camera.profile:=1920x1080x30 \
  infra_profile:=848x480x30 \
  align_depth.enable:=true \
  depth_module.emitter_enabled:=0"
