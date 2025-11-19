# Find MPI package
find_package(MPI REQUIRED)

# Static library target
add_library(ParallelABM STATIC
  src/Logger.cpp
  src/MPINode.cpp
  src/MPIWorker.cpp
  src/MPICoordinator.cpp
  src/EnvironmentSlurm.cpp
)

# Set C++ standard to C++20 (required for std::span)
target_compile_features(ParallelABM PUBLIC cxx_std_20)

# Include directories
target_include_directories(ParallelABM
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ParallelABM
  PUBLIC
    MPI::MPI_CXX
)

# Collect all public headers
file(GLOB PARALLELABM_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/ParallelABM/*.h")

# Set library properties
set_target_properties(ParallelABM PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER "${PARALLELABM_HEADERS}"
)

# Installation rules
install(TARGETS ParallelABM
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include/ParallelABM
)
