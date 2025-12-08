# Game of Life examples

# Build common library first (shared between CPU and GPU)
add_subdirectory(common)

add_subdirectory(cpu)

# Only build GPU examples if CUDA is enabled
if(CUDA)
  add_subdirectory(gpu)
endif()
