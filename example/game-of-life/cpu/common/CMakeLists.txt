# Game of Life CPU-specific common library
#
# CPU-specific shared code between CPU-based Game of Life examples (UI and time-measurement).

cmake_minimum_required(VERSION 3.14)

# Collect all source files from src directory
file(GLOB GAMEOFLIFE_CPU_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create static library for CPU-specific common code
add_library(game_of_life_cpu_common STATIC ${GAMEOFLIFE_CPU_COMMON_SOURCES})

# Set C++20 standard
target_compile_features(game_of_life_cpu_common PUBLIC cxx_std_20)

# Include directories for headers
target_include_directories(game_of_life_cpu_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against base common library and ParallelABM
target_link_libraries(game_of_life_cpu_common
  PUBLIC
    game_of_life_common
    ParallelABM
)
