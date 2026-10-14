# SBEG 461 Lab 4 -- C++ track (lab_tests, lab_bench) and the Step 3 RNEA benchmark (rnea_bench, both tracks).
# Build and test (from this folder):
#   cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
# or, from your Lab code folder:  python run_lab.py 4 --cpp
cmake_minimum_required(VERSION 3.16)
project(sbeg461_lab04 CXX)

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

# helpers.cpp: the provided helpers of starter.py (given code, do not edit)
sbeg_lab(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp")

# ---- C++ track, Step 3 table: lab_bench times your lab::rnea and mr::inverse_dynamics and writes the files
# ---- model.txt and states.txt that rnea_bench reads:   ./build/lab_bench [output folder]
add_executable(lab_bench bench_main.cpp helpers.cpp "${LAB_IMPL}.cpp")
target_include_directories(lab_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(lab_bench PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
target_link_libraries(lab_bench PRIVATE sbeg_mrcpp)

# ---- Step 3 (both tracks): the rnea_bench program = main.cpp + model_io.cpp + your rnea_starter.cpp.
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release                 (your rnea_starter.cpp)
#   cmake -S . -B build -DRNEA_SRC=<other .cpp file>                (build another RNEA implementation)
#   cmake --build build      ->  build/rnea_bench   (Windows: build\rnea_bench.exe)
# It uses the Eigen copy in cpp_support (no separate Eigen installation needed).
# Without CMake:  c++ -O3 -DNDEBUG -std=c++17 -I<eigen3 include dir> main.cpp model_io.cpp rnea_starter.cpp -o rnea_bench
set(RNEA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rnea_starter.cpp" CACHE FILEPATH "RNEA source file to build (default: your rnea_starter.cpp)")
add_executable(rnea_bench main.cpp model_io.cpp "${RNEA_SRC}")
target_include_directories(rnea_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(rnea_bench SYSTEM PRIVATE "${SBEG_CPP_SUPPORT}/third_party/eigen")
target_compile_definitions(rnea_bench PRIVATE EIGEN_MPL2_ONLY)

foreach(_sbeg_tgt lab_bench rnea_bench)
  target_compile_features(${_sbeg_tgt} PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(${_sbeg_tgt} PRIVATE /utf-8 /bigobj /EHsc /permissive- /W3)
  else()
    target_compile_options(${_sbeg_tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter $<$<CONFIG:Release>:-O3>)
  endif()
  # the programs go directly into the build folder for every generator (as lab_tests)
  set_target_properties(${_sbeg_tgt} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}")
endforeach()
