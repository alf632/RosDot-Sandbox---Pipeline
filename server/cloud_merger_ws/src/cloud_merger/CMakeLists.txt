cmake_minimum_required(VERSION 3.8)
project(cloud_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common filters)

add_executable(merge_node src/cloud_merger_node.cpp)
target_include_directories(merge_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(merge_node ${PCL_LIBRARIES})
ament_target_dependencies(merge_node rclcpp sensor_msgs message_filters tf2_ros tf2_sensor_msgs pcl_conversions)

install(TARGETS merge_node DESTINATION lib/${PROJECT_NAME})
ament_package()
