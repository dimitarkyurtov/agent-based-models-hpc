# Test 5: Game of Life tall grid (1200x300)
#
# Test application with 1200x300 grid, 100 timesteps, checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB TEST5_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(test5 ${TEST5_SOURCES})

# Set C++20 standard
target_compile_features(test5 PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(test5
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(test5
  PRIVATE
    test_common
)

# Set output directory
set_target_properties(test5 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
