# Test 2: Game of Life medium grid (500x500)
#
# Test application with 500x500 grid, 100 timesteps, checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB TEST2_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(test2 ${TEST2_SOURCES})

# Set C++20 standard
target_compile_features(test2 PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(test2
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(test2
  PRIVATE
    test_common
)

# Set output directory
set_target_properties(test2 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
