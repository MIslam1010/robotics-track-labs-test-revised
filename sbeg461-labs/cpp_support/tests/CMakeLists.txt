# Self-test of cpp_support: the modern_robotics C++ port against the Python library.
#   cmake -S tests -B build-tests -G Ninja && cmake --build build-tests && ./build-tests/mrcpp_tests
# (run from the cpp_support folder; regenerate the cases with:  python tests/make_mrcpp_cases.py)
cmake_minimum_required(VERSION 3.16)
project(sbeg_cpp_support_tests CXX)

set(SBEG_CPP_SUPPORT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "cpp_support folder")
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SBEGLab.cmake")

add_executable(mrcpp_tests test_mrcpp.cpp)
_sbeg_configure_target(mrcpp_tests)
target_compile_definitions(mrcpp_tests PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
target_link_libraries(mrcpp_tests PRIVATE sbeg_mrcpp sbeg_doctest_main)
set_target_properties(mrcpp_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME mrcpp_tests COMMAND mrcpp_tests)
