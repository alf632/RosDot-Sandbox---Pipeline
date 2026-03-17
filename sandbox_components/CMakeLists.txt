cmake_minimum_required(VERSION 3.8)
project(sandbox_components)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

# 1. Reprojector Component
add_library(camera_reprojector SHARED src/camera_reprojector.cpp)
target_compile_definitions(camera_reprojector PRIVATE "CAMERA_REPROJECTOR_BUILDING_DLL")
ament_target_dependencies(camera_reprojector rclcpp rclcpp_components sensor_msgs cv_bridge tf2_ros tf2_eigen Eigen3 OpenCV)
rclcpp_components_register_nodes(camera_reprojector "sandbox_components::CameraReprojector")

# 2. Merger Component
add_library(cloud_merger SHARED src/cloud_merger.cpp)
target_compile_definitions(cloud_merger PRIVATE "CLOUD_MERGER_BUILDING_DLL")
ament_target_dependencies(cloud_merger rclcpp rclcpp_components sensor_msgs cv_bridge OpenCV)
rclcpp_components_register_nodes(cloud_merger "sandbox_components::CloudMerger")

# 3. UDP Streamer Component
add_library(heightmap_streamer SHARED src/heightmap_streamer.cpp)
target_compile_definitions(heightmap_streamer PRIVATE "HEIGHTMAP_STREAMER_BUILDING_DLL")
ament_target_dependencies(heightmap_streamer rclcpp rclcpp_components sensor_msgs cv_bridge OpenCV)
rclcpp_components_register_nodes(heightmap_streamer "sandbox_components::HeightmapStreamer")

install(TARGETS camera_reprojector cloud_merger heightmap_streamer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
