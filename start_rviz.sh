xhost +local:docker
docker run --rm -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/Docker/Humble/dev_ws:/root/dev_ws" \
  --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
  --env="ROS_DOMAIN_ID=42" \
  --device=/dev/dri:/dev/dri \
  --name ubuntu-humble \
  --hostname ros2-humble \
  --net=host \
  --ipc=host \
  osrf/ros:humble-desktop-full
