cmake_minimum_required(VERSION 3.14)
project(MyLibProject VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add compiler warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Option to enable/disable testing
option(BUILD_TESTING "Build tests" ON)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(example)

# Add tests if enabled
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
