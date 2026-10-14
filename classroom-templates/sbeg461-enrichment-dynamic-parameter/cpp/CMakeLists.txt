# SBEG 461 enrichment lab "parameter identification" -- C++ track.  Build and test (from this folder):
#   cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
# or, from your Lab code folder:  cmake -S cpp -B build -G Ninja && cmake --build build && ./build/lab_tests
cmake_minimum_required(VERSION 3.16)
project(sbeg461_param_identification CXX)

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

# given code (do not edit): robot models, log helpers, NumPy/SciPy equivalents, provided functions of starter.py
sbeg_lab(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/robots.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/hwlog.cpp"
                 "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/provided.cpp")
