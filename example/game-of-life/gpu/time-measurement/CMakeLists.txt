# Game of Life GPU time measurement example
#
# Demonstrates the ParallelABM library usage with Conway's
# Game of Life simulation for GPU performance measurement using CUDA.

cmake_minimum_required(VERSION 3.14)

# Enable CUDA language support
enable_language(CUDA)

# Collect C++ source files
file(GLOB GAMEOFLIFE_GPU_TIME_CPP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Collect CUDA source files
file(GLOB GAMEOFLIFE_GPU_TIME_CUDA_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

# Create executable
add_executable(game_of_life_gpu_time
  ${GAMEOFLIFE_GPU_TIME_CPP_SOURCES}
  ${GAMEOFLIFE_GPU_TIME_CUDA_SOURCES})

# Set C++20 standard
target_compile_features(game_of_life_gpu_time PRIVATE cxx_std_20)

# Set CUDA architecture (adjust for your GPU)
set_property(TARGET game_of_life_gpu_time PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Include directories for local headers
target_include_directories(game_of_life_gpu_time
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link against required libraries
target_link_libraries(game_of_life_gpu_time
  PRIVATE
    game_of_life_common
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(game_of_life_gpu_time PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
