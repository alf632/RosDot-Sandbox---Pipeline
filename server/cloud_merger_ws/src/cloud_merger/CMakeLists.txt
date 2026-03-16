cmake_minimum_required(VERSION 3.8)
project(cloud_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

cmake_policy(SET CMP0074 NEW)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
# message_filters removed!
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common filters)

# Point this to your new C++ file
add_executable(dynamic_merger src/dynamic_cloud_merger.cpp)

target_include_directories(dynamic_merger PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(dynamic_merger ${PCL_LIBRARIES})

# message_filters removed from ament_target_dependencies
ament_target_dependencies(dynamic_merger 
  rclcpp 
  sensor_msgs 
  tf2_ros 
  tf2_sensor_msgs 
  pcl_conversions
)

install(TARGETS dynamic_merger DESTINATION lib/${PROJECT_NAME})
ament_package()
