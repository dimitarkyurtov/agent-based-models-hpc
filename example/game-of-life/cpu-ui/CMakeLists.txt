# Game of Life CPU example with ImGui UI
#
# This example demonstrates the ParallelABM library usage with Conway's
# Game of Life simulation rendered using ImGui and OpenGL.

cmake_minimum_required(VERSION 3.14)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Collect all source files from src directory
file(GLOB GAMEOFLIFE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(game_of_life_cpu_ui ${GAMEOFLIFE_SOURCES})

# Set C++20 standard
target_compile_features(game_of_life_cpu_ui PRIVATE cxx_std_20)

# Include directories for local headers
target_include_directories(game_of_life_cpu_ui
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against required libraries
target_link_libraries(game_of_life_cpu_ui
  PRIVATE
    ParallelABM
    OpenGL::GL
    glfw
    imgui
)

# Set output directory
set_target_properties(game_of_life_cpu_ui PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
