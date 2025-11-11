# Static library target
add_library(ParallelABM STATIC
  src/calculator.cpp
)

# Include directories
target_include_directories(ParallelABM
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library properties
set_target_properties(ParallelABM PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER include/ParallelABM/calculator.h
)

# Installation rules
install(TARGETS ParallelABM
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include/ParallelABM
)
