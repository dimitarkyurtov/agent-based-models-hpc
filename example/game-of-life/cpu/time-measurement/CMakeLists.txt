cmake_minimum_required(VERSION 3.14)

file(GLOB GAMEOFLIFE_TIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(game_of_life_cpu_time ${GAMEOFLIFE_TIME_SOURCES})

target_compile_features(game_of_life_cpu_time PRIVATE cxx_std_20)

target_include_directories(game_of_life_cpu_time
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(game_of_life_cpu_time
  PRIVATE
    game_of_life_cpu_common
)

set_target_properties(game_of_life_cpu_time PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
