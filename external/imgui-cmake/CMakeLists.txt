# ImGui library build configuration
cmake_minimum_required(VERSION 3.14)

project(imgui)

# Find OpenGL and GLFW
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# ImGui core sources from submodule
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../imgui)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Create ImGui library
add_library(imgui STATIC ${IMGUI_SOURCES})

# Disable static analysis for imgui (external dependency)
set_target_properties(imgui PROPERTIES
    CXX_CLANG_TIDY ""
    CXX_CPPCHECK ""
    CXX_CPPLINT ""
    CXX_INCLUDE_WHAT_YOU_USE ""
)

# Include directories for ImGui
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# Link dependencies
target_link_libraries(imgui PUBLIC
    OpenGL::GL
    glfw
)

# Set C++20 standard
target_compile_features(imgui PUBLIC cxx_std_20)
