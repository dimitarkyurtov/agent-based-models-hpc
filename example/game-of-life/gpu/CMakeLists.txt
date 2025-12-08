# Game of Life GPU examples
#
# This directory contains GPU-based Game of Life examples using CUDA.

cmake_minimum_required(VERSION 3.14)

# Verify CUDA is enabled (this subdirectory should only be included if CUDA is ON)
if(NOT CUDA)
  message(FATAL_ERROR "CUDA support is required for GPU examples but is disabled")
endif()

# Enable CUDA language support
enable_language(CUDA)

# Build time measurement application
add_subdirectory(time-measurement)