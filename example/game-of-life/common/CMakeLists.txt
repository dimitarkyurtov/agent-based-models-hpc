cmake_minimum_required(VERSION 3.14)

find_package(OpenSSL REQUIRED)

file(GLOB GAMEOFLIFE_COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(game_of_life_common STATIC ${GAMEOFLIFE_COMMON_SOURCES})

target_compile_features(game_of_life_common PUBLIC cxx_std_20)

target_include_directories(game_of_life_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(game_of_life_common
  PUBLIC
    ParallelABM
    OpenSSL::Crypto
)
