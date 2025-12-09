# Test 3: Game of Life large grid (1000x1000)
#
# Test application with 1000x1000 grid, 100 timesteps, checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB TEST3_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(test3 ${TEST3_SOURCES})

# Set C++20 standard
target_compile_features(test3 PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(test3
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(test3
  PRIVATE
    test_common
)

# Set output directory
set_target_properties(test3 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
