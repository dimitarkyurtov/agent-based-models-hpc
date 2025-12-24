cmake_minimum_required(VERSION 3.14)

file(GLOB GAMEOFLIFE_CPU_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(game_of_life_cpu_common STATIC ${GAMEOFLIFE_CPU_COMMON_SOURCES})

target_compile_features(game_of_life_cpu_common PUBLIC cxx_std_20)

target_include_directories(game_of_life_cpu_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(game_of_life_cpu_common
  PUBLIC
    game_of_life_common
    ParallelABM
)
