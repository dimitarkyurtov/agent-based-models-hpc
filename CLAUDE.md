# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParallelABM is a C++ library that abstracts the complexity of running agent-based model simulations in high-performance computing (HPC) environments. The library provides a simplified interface for distributed agent-based simulations on CPU and GPU (NVIDIA) architectures, handling parallelization and MPI communication automatically.

## Build Commands

### Standard Build
```bash
# Configure with compile commands for clang-tidy
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build
```

### CUDA Build
```bash
# Configure with CUDA support
cmake -B build -DCUDA=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build
```

### Testing
```bash
# Run all tests
cd build && ctest --output-on-failure

# Run tests with verbose output
cd build && ctest -V
```

### Docker Build Environment

A Docker container with CUDA 12.3.1 and OpenMPI 4.1.6 is provided for building GPU code without local NVIDIA hardware:

```bash
# Build and start the container
docker-compose up -d

# Enter the container
docker exec -it mpi-nvidia-dev bash

# Inside container: build with CUDA support
cmake -B build -DCUDA=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# Exit container
exit

# Stop container
docker-compose down
```

The container includes all required dependencies: CMake, OpenMPI with CUDA support, clang-format, clang-tidy, and development tools.

### Code Quality

**Formatting (clang-format):**
```bash
# Format all source files
find lib example tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) | xargs clang-format -i

# Check formatting without modifying files
find lib example tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) | xargs clang-format --dry-run --Werror
```

**Static Analysis (clang-tidy):**
```bash
# Run clang-tidy on C++ sources (excludes CUDA files and external dependencies)
find lib example -type f \( -name '*.cpp' \) ! -path '*/external/*' ! -path '*/imgui*' ! -path '*/gpu/*' ! -name '*CUDA*' | xargs clang-tidy -p build
```

## Architecture Overview

### Core Design Pattern

The library uses a **coordinator-worker MPI pattern** combined with **template-based polymorphism** for device abstraction:

1. **MPI Distribution Layer**: `MPICoordinator` (rank 0) partitions the simulation space into regions and distributes them to `MPIWorker` instances (rank 1+). Each process handles its local region independently.

2. **Device Execution Layer**: Users inherit from `Simulation<AgentT, ModelType>` and implement:
   - `SimulationCPU` for multi-threaded CPU execution (uses BS::thread_pool)
   - `SimulationCUDA` for GPU execution (when CUDA is enabled)

3. **User Model Layer**: Users define agent behavior by inheriting from:
   - `ModelCPU<AgentT>`: Override `ComputeInteractions()` with CPU interaction logic
   - `ModelCUDA<AgentT>`: Provide CUDA device functions for GPU execution

### Key Components

**Space and Regions:**
- `Space<AgentT>`: User-defined spatial organization (2D/3D grid, network, etc.)
- `Space::Region`: Partition of agents for distribution across MPI processes
- `LocalRegion<AgentT>`: The agents assigned to a specific MPI process
- `LocalSubRegion<AgentT>`: Further subdivision of local region for threading

**Execution Flow:**
1. `Simulation` constructor initializes MPI and creates coordinator/workers based on rank
2. `Start(timesteps)` begins execution:
   - Coordinator sends initial regions to workers
   - Each timestep: `LaunchModel()` executes model on local region
   - Workers send updated regions back to coordinator
   - Coordinator updates neighbors and sends to workers for next step
   - `OnTimeStepCompleted()` hook called (for rendering, checkpointing, etc.)

**Environment Abstraction:**
- `Environment`: Abstract interface for querying CPU/GPU resources
- `EnvironmentSlurm`: Reads SLURM environment variables for HPC clusters
- `LocalEnvironment`: For local development (reads from constructor arguments)

### Template Requirements

Agent types must be:
- Default constructible (for MPI receive operations)
- Copy constructible and copy assignable (for MPI send/receive)
- Preferably trivially copyable for optimal MPI performance

### Implementation Patterns

**Header Organization:**
- Interface headers (`.h`) contain declarations and documentation
- Template implementations live in `.inl` files included at the end of headers
- Source files (`.cpp`) are excluded from CMake if they're templated

**CMake Filtering:**
The library CMakeLists.txt excludes template-based files from compilation:
```cmake
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*LocalRegion\\.cpp$")
list(FILTER PARALLELABM_SOURCES EXCLUDE REGEX ".*SimulationCPU\\.cpp$")
```

## Example Structure

The repository includes Game of Life as a reference implementation:
- `example/game-of-life/common/`: Shared code (Cell agent, GameOfLifeSpace)
- `example/game-of-life/cpu/`: CPU implementations with UI and time measurement variants
- `example/game-of-life/gpu/`: GPU implementation (requires CUDA=ON)

### Running Examples

After building, executables are in `build/bin/`. Examples are run using MPI:

```bash
# CPU time measurement: <num_processes> × <num_threads> parallelism
mpirun -n 4 ./build/bin/game_of_life_cpu_time 2 predefined
# 4 MPI processes, 2 CPU threads each

# GPU example (requires CUDA build)
mpirun -n 2 ./build/bin/game_of_life_gpu_time
# 2 MPI processes, each using 1 GPU

# UI example (single process recommended for visualization)
mpirun -n 1 ./build/bin/game_of_life_cpu_ui
```

Each example's README provides specific usage instructions and command-line arguments.

## Testing Strategy

Tests run simulations against pre-recorded checkpoints:
- Each test runs a specific grid configuration (100x100, 500x500, 1000x1000, etc.)
- Checkpoints are taken every 50 timesteps
- Simulation state is hashed and compared against expected values
- This validates correctness across different configurations and MPI process counts

## Dependencies

**Required:**
- CMake 3.14+
- C++17 compiler (library uses C++20 for std::span internally)
- OpenMPI
- clang-format, clang-tidy

**Optional:**
- NVIDIA CUDA Toolkit (for GPU support, enable with `-DCUDA=ON`)

**Bundled External:**
- BS::thread_pool (header-only, in `external/thread-pool/`)
- imgui (for UI examples, in `external/imgui-cmake/`)

## Common Pitfalls

**Agent Type Design:**
- Ensure agents are trivially copyable when possible for MPI efficiency
- Avoid virtual functions in agent types (breaks MPI serialization)
- If agents need complex state, consider separating data from behavior

**MPI Communication:**
- Neighbor exchange happens automatically between timesteps
- Coordinator (rank 0) manages the full space state
- Workers only see their local region plus neighbors

**Threading vs MPI:**
- MPI distributes across nodes/processes (coarse-grained parallelism)
- CPU threads divide work within a process (fine-grained parallelism)
- Thread count is determined by `Environment::GetNumberOfCPUCores()`

**CUDA Compilation:**
- CUDA files (`.cu`, `.cuh`) are only compiled when `-DCUDA=ON` is set
- Guard CUDA-specific includes with `#ifdef PARALLELABM_CUDA_ENABLED`
- CPU and GPU code paths are separate but use the same `Simulation` interface
