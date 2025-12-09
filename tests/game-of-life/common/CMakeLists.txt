# Test common library
#
# Shared code for Game of Life test applications.

cmake_minimum_required(VERSION 3.14)

# Find OpenSSL package for SHA-256 hashing
find_package(OpenSSL REQUIRED)

# Collect all source files from src directory
file(GLOB TEST_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create static library for common test code
add_library(test_common STATIC ${TEST_COMMON_SOURCES})

# Set C++20 standard
target_compile_features(test_common PUBLIC cxx_std_20)

# Include directories for headers
target_include_directories(test_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENSSL_INCLUDE_DIR}
)

# Link against ParallelABM library and OpenSSL
target_link_libraries(test_common
  PUBLIC
    ParallelABM
    OpenSSL::Crypto
)
