# Game of Life Test Suite
#
# Test applications for verifying Game of Life simulation with different grid configurations.
# Each test runs 100 timesteps with checkpoints every 10 steps.

cmake_minimum_required(VERSION 3.14)

# Add common test library
add_subdirectory(common)

# Add test applications
add_subdirectory(test1)  # 100x100 grid
add_subdirectory(test2)  # 500x500 grid
add_subdirectory(test3)  # 1000x1000 grid
add_subdirectory(test4)  # 800x400 grid (rectangular)
add_subdirectory(test5)  # 1200x300 grid (wide)
