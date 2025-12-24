cmake_minimum_required(VERSION 3.14)

if(NOT CUDA)
  message(FATAL_ERROR "CUDA support is required for GPU examples but is disabled")
endif()

enable_language(CUDA)

add_subdirectory(time-measurement)