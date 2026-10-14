# SBEG 361 Lab 6 -- C++ track.  Build and test (from this folder):
#   cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
# or, from your Lab code folder:  python run_lab.py 6 --cpp
cmake_minimum_required(VERSION 3.16)
project(sbeg361_lab06 CXX)

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

sbeg_lab()

# ---- Part B (both tracks): the fk_compare program = main.cpp + robot_data.cpp + your fk_starter.cpp.
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   ./build/fk_compare ../data/q_samples.csv .        (Visual Studio generator: build\Release\fk_compare.exe)
# It uses the Eigen copy in cpp_support (no separate Eigen installation needed).
set(FK_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fk_starter.cpp" CACHE FILEPATH "Part B: the FK file built into fk_compare")
add_executable(fk_compare main.cpp robot_data.cpp "${FK_SOURCE}")
target_include_directories(fk_compare PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(fk_compare SYSTEM PRIVATE "${SBEG_CPP_SUPPORT}/third_party/eigen")
target_compile_definitions(fk_compare PRIVATE EIGEN_MPL2_ONLY)
target_compile_features(fk_compare PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(fk_compare PRIVATE /utf-8 /EHsc /permissive- /W3)
else()
  target_compile_options(fk_compare PRIVATE -Wall -Wextra)
endif()
