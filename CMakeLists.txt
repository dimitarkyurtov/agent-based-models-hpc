cmake_minimum_required(VERSION 3.14)

option(CUDA "Enable CUDA support for GPU simulations" OFF)

if(CUDA)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
endif()

if(CUDA)
  project(ParallelABMProject VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
  project(ParallelABMProject VERSION 1.0.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_TESTING "Build tests" ON)

find_package(Threads REQUIRED)

add_subdirectory(external/imgui-cmake)

add_library(ThreadPool INTERFACE)
target_include_directories(ThreadPool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/thread-pool/include)
target_link_libraries(ThreadPool INTERFACE Threads::Threads)

add_subdirectory(lib)
add_subdirectory(example)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
