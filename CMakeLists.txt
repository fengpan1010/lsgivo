cmake_minimum_required(VERSION 3.0.2)
project(lsgivo)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package (OpenCV 3 REQUIRED)
find_package (Eigen3 REQUIRED)
find_package (PCL REQUIRED)
find_package (Sophus REQUIRED)
find_package (yaml-cpp REQUIRED)

add_definitions( ${PCL_DEFINITIONS} )

find_package (catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    message_filters
    cv_bridge
    image_transport
    tf2_ros
    )

catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES lsgivo
    #  CATKIN_DEPENDS roscpp rospy std_msgs
    #  DEPENDS system_lib
    )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    # include
    ${catkin_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/"
    )

add_executable(vo
    src/main.cpp
    src/capture.cpp
    src/triangulation.cpp)
target_link_libraries(vo
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${Sophus_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${PCL_LIBRARIES})


