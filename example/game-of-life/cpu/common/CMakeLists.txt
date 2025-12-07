# Game of Life common library
#
# Shared code between CPU-based Game of Life examples (UI and time-measurement).

cmake_minimum_required(VERSION 3.14)

# Find OpenSSL package for SHA-256 hashing
find_package(OpenSSL REQUIRED)

# Collect all source files from src directory
file(GLOB GAMEOFLIFE_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create static library for common code
add_library(game_of_life_common STATIC ${GAMEOFLIFE_COMMON_SOURCES})

# Set C++20 standard
target_compile_features(game_of_life_common PUBLIC cxx_std_20)

# Include directories for headers
target_include_directories(game_of_life_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENSSL_INCLUDE_DIR}
)

# Link against ParallelABM library and OpenSSL
target_link_libraries(game_of_life_common
  PUBLIC
    ParallelABM
    OpenSSL::Crypto
)
