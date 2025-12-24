# Find MPI package
find_package(MPI REQUIRED)

file(GLOB PARALLELABM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*MPIStub\\.cpp$")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*LocalRegion\\.cpp$")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*SimulationCPU\\.cpp$")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*MPIWorker\\.cpp$")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*MPICoordinator\\.cpp$")

if(CUDA)
  find_package(CUDAToolkit REQUIRED)
  file(GLOB PARALLELABM_CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
  list(APPEND PARALLELABM_SOURCES ${PARALLELABM_CUDA_SOURCES})
endif()

add_library(ParallelABM STATIC ${PARALLELABM_SOURCES})

target_compile_features(ParallelABM PUBLIC cxx_std_20)

if(CUDA)
  set_target_properties(ParallelABM PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
  target_compile_definitions(ParallelABM PUBLIC PARALLELABM_CUDA_ENABLED)
endif()

target_include_directories(ParallelABM
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(ParallelABM
  PUBLIC
    MPI::MPI_CXX
    ThreadPool
)

if(CUDA)
  target_link_libraries(ParallelABM PUBLIC CUDA::cudart)
endif()

file(GLOB PARALLELABM_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/ParallelABM/*.h")

set_target_properties(ParallelABM PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER "${PARALLELABM_HEADERS}"
)

install(TARGETS ParallelABM
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include/ParallelABM
)
