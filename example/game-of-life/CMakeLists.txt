add_subdirectory(common)

add_subdirectory(cpu)

if(CUDA)
  add_subdirectory(gpu)
endif()
