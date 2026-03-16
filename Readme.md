calibration

 - start the realsense sensors with raspi/calibrate.sh to disable the IR Emitter and turn on infrared and rgb feeds.

 - layout apriltag(s) and publish their position towards sandbox_origin (the center) with something like:
ros2 run tf2_ros static_transform_publisher 3.0 2.0 0.5 0 0 0 tag36h11:0 sandbox_origin
e.g. how to move from the tags center to the sandbox's center

 - recognize apriltag(s) and get the tronsform from sandbox_origin to the cameras. automated in server/sensor_calibration.sh

 - remove the static transform from the tag to sandbox_origin. it will be replace with the recognized transforms from sandbox_origin to the cameras


