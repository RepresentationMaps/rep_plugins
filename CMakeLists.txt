cmake_minimum_required(VERSION 3.0.2)
project(rep_plugins)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
set(DISABLE_PAL_FLAGS 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  pluginlib
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros
  roscpp
  tf
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  geometry_msgs
  hri
  cv_bridge
  vision_msgs
  chatgpt_ros
  message_generation
)

find_package (Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(yaml-cpp REQUIRED)
pkg_search_module(IlmBase REQUIRED OpenEXR)
include_directories(
    include
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenVDB_INCLUDE_DIR}
    ${IlmBase_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
)

add_service_files(
  FILES
  ResetContext.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs 
)

catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
        geometry_msgs
        pcl_ros
        pcl_conversions
        pluginlib
        roscpp
        sensor_msgs
        std_msgs
        tf2_ros
        tf2_geometry_msgs
        visualization_msgs
        hri
        message_generation
    DEPENDS
        OpenCV
        TBB
        OpenVDB
        PCL
        Boost
	    IlmBase
        yaml-cpp
)

add_executable(context_republisher
  src/context_republisher.cpp
)
target_link_libraries(context_republisher
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${OpenVDB_LIBRARIES}
  ${TBB_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_library(${PROJECT_NAME}
            src/rep_plugin_base.cpp
            src/agents_gaze_plugin.cpp
            src/simple_faces_plugin.cpp
            src/object_detection_3d_plugin.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenVDB_LIBRARIES}
  ${IlmBase_LIBRARIES}
  ${catkin_LIBRARIES}
  ${TBB_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

