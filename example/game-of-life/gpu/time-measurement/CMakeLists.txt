cmake_minimum_required(VERSION 3.14)

enable_language(CUDA)

file(GLOB GAMEOFLIFE_GPU_TIME_CPP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

file(GLOB GAMEOFLIFE_GPU_TIME_CUDA_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_executable(game_of_life_gpu_time
  ${GAMEOFLIFE_GPU_TIME_CPP_SOURCES}
  ${GAMEOFLIFE_GPU_TIME_CUDA_SOURCES})

target_compile_features(game_of_life_gpu_time PRIVATE cxx_std_20)

set_property(TARGET game_of_life_gpu_time PROPERTY CUDA_STANDARD 17)
set_property(TARGET game_of_life_gpu_time PROPERTY CUDA_STANDARD_REQUIRED ON)

set_property(TARGET game_of_life_gpu_time PROPERTY CUDA_ARCHITECTURES 75 80 86)

target_include_directories(game_of_life_gpu_time
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(game_of_life_gpu_time
  PRIVATE
    game_of_life_common
    ${CUDA_LIBRARIES}
)

set_target_properties(game_of_life_gpu_time PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
