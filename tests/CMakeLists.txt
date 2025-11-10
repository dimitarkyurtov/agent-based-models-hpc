include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
include(GoogleTest)

# Test executable
add_executable(mylib_tests
  test_calculator.cpp
)

# Link against the library and Google Test
target_link_libraries(mylib_tests
  PRIVATE
    mylib
    GTest::gtest_main
)

# Discover tests
gtest_discover_tests(mylib_tests)
