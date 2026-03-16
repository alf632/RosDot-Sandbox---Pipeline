cmake_minimum_required(VERSION 3.8)
project(cloud_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Create the executable
add_executable(dynamic_cloud_merger src/dynamic_cloud_merger.cpp)
add_executable(heightmap_streamer src/heightmap_streamer.cpp)
ament_target_dependencies(heightmap_streamer rclcpp sensor_msgs cv_bridge)

# Target dependencies and linking
ament_target_dependencies(dynamic_cloud_merger
  rclcpp
  sensor_msgs
  cv_bridge
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  pcl_ros
)
ament_target_dependencies(heightmap_streamer rclcpp sensor_msgs cv_bridge)

target_link_libraries(dynamic_cloud_merger
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
target_link_libraries(heightmap_streamer ${OpenCV_LIBRARIES})

# Install rules
install(TARGETS
  dynamic_cloud_merger
  heightmap_streamer
  DESTINATION lib/${PROJECT_NAME}
)

# (Optional) Install a config folder if you create yaml files later
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
