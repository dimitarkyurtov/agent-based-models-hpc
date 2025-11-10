# Example application
add_executable(example_app main.cpp)

# Link against the library
target_link_libraries(example_app PRIVATE mylib)

# Installation rule
install(TARGETS example_app DESTINATION bin)
