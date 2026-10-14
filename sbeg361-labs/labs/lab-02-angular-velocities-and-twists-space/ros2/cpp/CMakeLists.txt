# SBEG 361 Lab 2, Part B -- optional C++ (rclcpp) version of nav_tf_broadcaster.py (ROS 2 Jazzy, ament_cmake).
# Not auto-tested. Build it in a colcon workspace: see the comment at the top of src/nav_tf_broadcaster.cpp.
cmake_minimum_required(VERSION 3.16)
project(sbeg361_nav_tf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
# JSON reader: the system package (Ubuntu: sudo apt install nlohmann-json3-dev) or, if it is not installed, the copy in
# the course folder cpp_support/third_party (searched in the folders above the REAL location of this package, so the
# colcon symlink ~/ros2_ws/src/sbeg361_nav_tf -> <Lab code folder>/ros2/cpp works).
find_package(nlohmann_json 3 QUIET)
if(NOT nlohmann_json_FOUND)
  get_filename_component(_sbeg_dir "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
  set(_sbeg_json "")
  foreach(_sbeg_level RANGE 8)
    foreach(_sbeg_try "${_sbeg_dir}/cpp_support/third_party" "${_sbeg_dir}/0 Course resources/cpp_support/third_party")
      if(NOT _sbeg_json AND EXISTS "${_sbeg_try}/nlohmann/json.hpp")
        set(_sbeg_json "${_sbeg_try}")
      endif()
    endforeach()
    get_filename_component(_sbeg_dir "${_sbeg_dir}" DIRECTORY)
  endforeach()
  if(NOT _sbeg_json)
    message(FATAL_ERROR "nlohmann/json not found: sudo apt install nlohmann-json3-dev, or keep the course folder "
                        "cpp_support next to (or above) your lab folders.")
  endif()
  add_library(sbeg_nlohmann_json INTERFACE)
  target_include_directories(sbeg_nlohmann_json SYSTEM INTERFACE "${_sbeg_json}")
  add_library(nlohmann_json::nlohmann_json ALIAS sbeg_nlohmann_json)
  message(STATUS "nlohmann/json: using the copy in ${_sbeg_json}")
endif()

# Default data folder: the Lab code folder two levels above the REAL location of this package (nav_tree.json,
# tool_track.csv); the symlink is resolved first so that ~/ros2_ws/src/sbeg361_nav_tf -> <Lab code folder>/ros2/cpp works.
get_filename_component(_sbeg_pkg "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(SBEG_LAB_DIR "${_sbeg_pkg}/../.." ABSOLUTE)

add_executable(nav_tf_broadcaster src/nav_tf_broadcaster.cpp)
target_compile_definitions(nav_tf_broadcaster PRIVATE "SBEG_LAB_DIR=\"${SBEG_LAB_DIR}\"")
target_link_libraries(nav_tf_broadcaster nlohmann_json::nlohmann_json)
ament_target_dependencies(nav_tf_broadcaster rclcpp geometry_msgs tf2_ros)

install(TARGETS nav_tf_broadcaster DESTINATION lib/${PROJECT_NAME})
ament_package()
