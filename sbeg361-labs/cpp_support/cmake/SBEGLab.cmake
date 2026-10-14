# SBEGLab.cmake -- shared CMake logic for the SBEG 361 / 461 / 46X C++ labs.
#
# A lab's CMakeLists.txt finds this file (see the lab template below), includes it and calls sbeg_lab():
#
#   cmake_minimum_required(VERSION 3.16)
#   project(sbeg361_lab01 CXX)
#   ... support-dir discovery (copy verbatim from any lab) ...
#   include("${SBEG_CPP_SUPPORT}/cmake/SBEGLab.cmake")
#   sbeg_lab()
#
# sbeg_lab([NAME lab_tests] [SOURCES extra.cpp ...])
#   builds the executable `lab_tests` from
#     lab.hpp + (starter.cpp | solution.cpp, chosen by -DLAB_IMPL=starter|solution, default starter)
#     + test_lab.cpp + any extra SOURCES + the shared mrcpp library + the doctest main.
#   Defines SBEG_LAB_DIR="<this lab's cpp folder>" so tests find cpp_cases.json and ../data/ from any build folder.
#
# Options / cache variables:
#   LAB_IMPL          starter (default) | solution
#   SBEG_CPP_SUPPORT  path of cpp_support (found automatically; may be given with -D or as an environment variable)

include_guard(GLOBAL)

if(NOT DEFINED SBEG_CPP_SUPPORT OR NOT EXISTS "${SBEG_CPP_SUPPORT}/include/sbeg/mrcpp.hpp")
  get_filename_component(SBEG_CPP_SUPPORT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()
set(SBEG_CPP_SUPPORT "${SBEG_CPP_SUPPORT}" CACHE PATH "Folder of the shared C++ support (cpp_support)" FORCE)

set(LAB_IMPL "starter" CACHE STRING "Which implementation to test: starter (your work) or solution")
set_property(CACHE LAB_IMPL PROPERTY STRINGS starter solution)

# Single-configuration generators (Ninja, Makefiles): optimise by default but keep debug information.
get_property(_sbeg_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _sbeg_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler settings shared by mrcpp and the lab.
function(_sbeg_configure_target tgt)
  target_compile_features(${tgt} PUBLIC cxx_std_17)
  if(MSVC)
    # /utf-8: source files are UTF-8; /bigobj: Eigen + doctest objects are large; /EHsc: C++ exceptions;
    # /permissive-: standard conformance (same rules as GCC/Clang). _USE_MATH_DEFINES is NOT relied upon.
    target_compile_options(${tgt} PRIVATE /utf-8 /bigobj /EHsc /permissive- /W3)
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  else()
    # -Wno-unused-parameter: starter functions ignore their arguments until they are written.
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

function(_sbeg_add_support_includes tgt)
  # SYSTEM: no warnings from the vendored libraries.
  target_include_directories(${tgt} SYSTEM PUBLIC
    "${SBEG_CPP_SUPPORT}/third_party/eigen"
    "${SBEG_CPP_SUPPORT}/third_party")
  target_include_directories(${tgt} PUBLIC "${SBEG_CPP_SUPPORT}/include")
  target_compile_definitions(${tgt} PUBLIC EIGEN_MPL2_ONLY)
endfunction()

# The shared modern_robotics port (built once per build folder).
if(NOT TARGET sbeg_mrcpp)
  add_library(sbeg_mrcpp STATIC "${SBEG_CPP_SUPPORT}/src/mrcpp.cpp")
  _sbeg_add_support_includes(sbeg_mrcpp)
  _sbeg_configure_target(sbeg_mrcpp)
endif()

if(NOT TARGET sbeg_doctest_main)
  add_library(sbeg_doctest_main STATIC "${SBEG_CPP_SUPPORT}/src/doctest_main.cpp")
  _sbeg_add_support_includes(sbeg_doctest_main)
  _sbeg_configure_target(sbeg_doctest_main)
endif()

function(sbeg_lab)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES" ${ARGN})
  if(NOT ARG_NAME)
    set(ARG_NAME lab_tests)
  endif()
  set(_dir "${CMAKE_CURRENT_SOURCE_DIR}")

  if(NOT LAB_IMPL STREQUAL "starter" AND NOT LAB_IMPL STREQUAL "solution")
    message(FATAL_ERROR "LAB_IMPL must be 'starter' or 'solution' (got '${LAB_IMPL}')")
  endif()
  set(_impl "${_dir}/${LAB_IMPL}.cpp")
  if(NOT EXISTS "${_impl}")
    message(FATAL_ERROR "LAB_IMPL=${LAB_IMPL} but ${_impl} does not exist "
                        "(students only have starter.cpp: configure with -DLAB_IMPL=starter).")
  endif()
  foreach(_f lab.hpp test_lab.cpp)
    if(NOT EXISTS "${_dir}/${_f}")
      message(FATAL_ERROR "missing ${_dir}/${_f}")
    endif()
  endforeach()

  add_executable(${ARG_NAME} "${_dir}/lab.hpp" "${_impl}" "${_dir}/test_lab.cpp" ${ARG_SOURCES})
  _sbeg_configure_target(${ARG_NAME})
  target_include_directories(${ARG_NAME} PRIVATE "${_dir}")
  target_compile_definitions(${ARG_NAME} PRIVATE "SBEG_LAB_DIR=\"${_dir}\"" "SBEG_LAB_IMPL=\"${LAB_IMPL}\"")
  target_link_libraries(${ARG_NAME} PRIVATE sbeg_mrcpp sbeg_doctest_main)
  # Put the program directly in the build folder for every generator (also Visual Studio: build/lab_tests.exe).
  set_target_properties(${ARG_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}")
  message(STATUS "SBEG lab: ${ARG_NAME} uses ${LAB_IMPL}.cpp; support: ${SBEG_CPP_SUPPORT}")
endfunction()
