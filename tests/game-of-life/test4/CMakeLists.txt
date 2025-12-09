# Test 4: Game of Life wide grid (800x400)
#
# Test application with 800x400 grid, 100 timesteps, checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB TEST4_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(test4 ${TEST4_SOURCES})

# Set C++20 standard
target_compile_features(test4 PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(test4
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(test4
  PRIVATE
    test_common
)

# Set output directory
set_target_properties(test4 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
