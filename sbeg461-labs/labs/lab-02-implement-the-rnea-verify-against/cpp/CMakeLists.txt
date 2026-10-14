# SBEG 461 Lab 2 -- C++ track.  Build and test (from this folder):
#   cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
# or, from your Lab code folder:  python run_lab.py 2 --cpp
cmake_minimum_required(VERSION 3.16)
project(sbeg461_lab02 CXX)

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
include("${SBEG_CPP_SUPPORT}/cmake/SBEGLab.cmake")
# ---- end of the fixed block

sbeg_lab(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pinlite.cpp")   # given Pinocchio stand-in (do not edit)

# ---- given program (Checkpoint 6 and optional timing): ./build/torque_profiles [--time] (do not edit)
add_executable(torque_profiles "${CMAKE_CURRENT_SOURCE_DIR}/torque_profiles.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_IMPL}.cpp"
               "${CMAKE_CURRENT_SOURCE_DIR}/pinlite.cpp")
_sbeg_configure_target(torque_profiles)
target_include_directories(torque_profiles PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(torque_profiles PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
target_link_libraries(torque_profiles PRIVATE sbeg_mrcpp)
foreach(_cfg "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
  set_target_properties(torque_profiles PROPERTIES RUNTIME_OUTPUT_DIRECTORY${_cfg} "${CMAKE_BINARY_DIR}")
endforeach()
