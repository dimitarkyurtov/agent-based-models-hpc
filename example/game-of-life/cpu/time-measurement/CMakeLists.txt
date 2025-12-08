# Game of Life CPU time measurement example
#
# Demonstrates the ParallelABM library usage with Conway's
# Game of Life simulation for performance measurement without visualization.

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB GAMEOFLIFE_TIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(game_of_life_cpu_time ${GAMEOFLIFE_TIME_SOURCES})

# Set C++20 standard
target_compile_features(game_of_life_cpu_time PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(game_of_life_cpu_time
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(game_of_life_cpu_time
  PRIVATE
    game_of_life_cpu_common
)

# Set output directory
set_target_properties(game_of_life_cpu_time PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
