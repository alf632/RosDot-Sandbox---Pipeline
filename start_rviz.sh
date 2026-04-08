xhost +local:docker
docker run --rm -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
  --env="ROS_DOMAIN_ID=42" \
  --volume="${PWD}/observer.rviz:/opt/observer.rviz" \
  --device=/dev/dri:/dev/dri \
  --name ubuntu-jazzy \
  --hostname ros2-jazzy \
  --net=host \
  --ipc=host \
  osrf/ros:jazzy-desktop-full
