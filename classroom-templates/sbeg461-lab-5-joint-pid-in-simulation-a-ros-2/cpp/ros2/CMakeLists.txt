# SBEG 461 Lab 5, Part D -- C++ ROS 2 controller node (optional; needs ROS 2 Jazzy, not needed for the auto-checks).
# Build from your Lab code folder, in a terminal where ROS 2 is sourced:
#   source /opt/ros/jazzy/setup.bash
#   cmake -S cpp/ros2 -B build_ros && cmake --build build_ros
#   ./build_ros/joint_pid_node --ros-args -p plant:=arm_j2 -p wn:=20.0 -p reference:=step -p q_target:=-0.3
# The node is linked with YOUR ../starter.cpp (JointPID, pd_gains and the provided functions).
cmake_minimum_required(VERSION 3.16)
project(sbeg461_lab05_ros2 CXX)

# ---- find cpp_support (do not edit): -DSBEG_CPP_SUPPORT=..., environment variable SBEG_CPP_SUPPORT,
# ---- or a folder "cpp_support" or "0 Course resources/cpp_support" in one of the 6 folders above this one.
if(NOT SBEG_CPP_SUPPORT AND DEFINED ENV{SBEG_CPP_SUPPORT})
  set(SBEG_CPP_SUPPORT "$ENV{SBEG_CPP_SUPPORT}")
endif()
if(NOT SBEG_CPP_SUPPORT)
  set(_sbeg_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  foreach(_sbeg_level RANGE 6)
    foreach(_sbeg_try "${_sbeg_dir}/cpp_support" "${_sbeg_dir}/0 Course resources/cpp_support")
      if(NOT SBEG_CPP_SUPPORT AND EXISTS "${_sbeg_try}/cmake/SBEGLab.cmake")
        set(SBEG_CPP_SUPPORT "${_sbeg_try}")
      endif()
    endforeach()
    get_filename_component(_sbeg_dir "${_sbeg_dir}" DIRECTORY)
  endforeach()
endif()
if(NOT SBEG_CPP_SUPPORT OR NOT EXISTS "${SBEG_CPP_SUPPORT}/cmake/SBEGLab.cmake")
  message(FATAL_ERROR "cpp_support not found. Copy the course folder 'cpp_support' next to your lab folders "
                      "or configure with -DSBEG_CPP_SUPPORT=<path to cpp_support>.")
endif()
# ---- end of the fixed block

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set(LAB_IMPL "starter" CACHE STRING "Which implementation of the lab functions to link (starter = your work)")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

get_filename_component(LAB_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_executable(joint_pid_node joint_pid_node.cpp "${LAB_CPP_DIR}/${LAB_IMPL}.cpp")
target_include_directories(joint_pid_node SYSTEM PRIVATE
  "${SBEG_CPP_SUPPORT}/third_party/eigen" "${SBEG_CPP_SUPPORT}/third_party")
target_include_directories(joint_pid_node PRIVATE "${SBEG_CPP_SUPPORT}/include" "${LAB_CPP_DIR}")
# SBEG_LAB_DIR: the lab's cpp folder, so that load_plant / load_knee_swing find ../data from any build folder.
target_compile_definitions(joint_pid_node PRIVATE EIGEN_MPL2_ONLY "SBEG_LAB_DIR=\"${LAB_CPP_DIR}\"")
if(NOT MSVC)
  target_compile_options(joint_pid_node PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
ament_target_dependencies(joint_pid_node rclcpp rcl_interfaces sensor_msgs std_msgs)
