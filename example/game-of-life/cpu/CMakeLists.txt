# Game of Life CPU examples
#
# This directory contains CPU-based Game of Life examples with shared common code.

cmake_minimum_required(VERSION 3.14)

# Build common library first
add_subdirectory(common)

# Build UI application
add_subdirectory(ui)

# Build time measurement application
add_subdirectory(time-measurement)
