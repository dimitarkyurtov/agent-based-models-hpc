# Test 1: Game of Life small grid (100x100)
#
# Test application with 100x100 grid, 100 timesteps, checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB TEST1_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(test1 ${TEST1_SOURCES})

# Set C++20 standard
target_compile_features(test1 PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(test1
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(test1
  PRIVATE
    test_common
)

# Set output directory
set_target_properties(test1 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
