cmake_minimum_required(VERSION 3.14)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

file(GLOB GAMEOFLIFE_UI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(game_of_life_cpu_ui ${GAMEOFLIFE_UI_SOURCES})

target_compile_features(game_of_life_cpu_ui PRIVATE cxx_std_20)

if(APPLE)
  target_compile_options(game_of_life_cpu_ui PRIVATE -Wno-deprecated-declarations)
endif()

target_include_directories(game_of_life_cpu_ui
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(game_of_life_cpu_ui
  PRIVATE
    game_of_life_cpu_common
    OpenGL::GL
    glfw
    imgui
)

set_target_properties(game_of_life_cpu_ui PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
