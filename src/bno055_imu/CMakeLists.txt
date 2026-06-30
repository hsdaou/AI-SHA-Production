cmake_minimum_required(VERSION 3.8)
project(bno055_imu)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BNO055Data.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python modules directory
install(DIRECTORY bno055_imu/
  DESTINATION lib/python3.12/site-packages/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

# Install Python executables
install(PROGRAMS
  bno055_imu/bno055_node.py
  bno055_imu/imu_bridge.py
  bno055_imu/imu_pose_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Create symlinks without .py extension for ROS 2 naming convention
install(CODE "
  execute_process(COMMAND ln -sf bno055_node.py bno055_node
    WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  execute_process(COMMAND ln -sf imu_bridge.py imu_bridge
    WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  execute_process(COMMAND ln -sf imu_pose_publisher.py imu_pose_publisher
    WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
")

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
