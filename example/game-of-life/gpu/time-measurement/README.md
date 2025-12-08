# Game of Life GPU Time Measurement Example

CUDA-based implementation of Conway's Game of Life using the ParallelABM library for GPU performance measurement with hybrid MPI+GPU parallelization.

## Overview

This example demonstrates how to implement a GPU-accelerated agent-based model using the ParallelABM library's `SimulationCUDA` and `ModelCUDA` interfaces. It combines MPI for distributed computing across nodes with CUDA for GPU acceleration within each node, enabling scalable high-performance simulations.

The implementation shares common code (Cell, GameOfLifeSpace) with the CPU examples, demonstrating code reuse across different compute backends.

## Hybrid MPI+GPU Architecture

The ParallelABM framework orchestrates computation across multiple levels:

1. **MPI Level**: Distributes the simulation grid across multiple MPI ranks (typically one per node)
2. **GPU Level**: Each MPI rank can utilize multiple GPUs to process its assigned region
3. **Grid Partitioning**: The global grid is split into horizontal bands, with boundary cells exchanged between ranks via MPI

This hybrid approach enables efficient scaling across GPU clusters:
- Inter-node parallelism via MPI
- Intra-node parallelism via multi-GPU
- Fine-grained parallelism within each GPU via CUDA threads

## Implementation Details

### GameOfLifeModel (ModelCUDA)

- Implements Game of Life rules as CUDA kernels
- Uses constant memory for grid dimensions
- One thread per cell for maximum parallelism
- Handles toroidal boundary conditions (wrap-around grid)
- Processes local agents and neighbor cells received from adjacent MPI ranks

### GameOfLifeSimulation (SimulationCUDA)

- Extends `SimulationCUDA<Cell>` for GPU execution
- Manages CUDA memory and kernel launches via the framework
- Coordinates MPI communication for boundary exchange
- Supports multi-GPU execution within each MPI rank

### Configuration

- Grid size: 1000x1000 cells
- Timesteps: 1000
- Initial density: 30% alive cells
- Supports predefined patterns (gliders, oscillators) or random initialization

## Building

The example is built as part of the main CMake configuration:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
cmake --build . --target game_of_life_gpu_time
```

## Usage

### Single Node, Single GPU

```bash
./bin/game_of_life_gpu_time
```

### Single Node, Multiple GPUs

```bash
# Use 4 GPUs on the local node
./bin/game_of_life_gpu_time 4
```

### Multi-Node with MPI

```bash
# Run on 2 nodes, 1 GPU per node
mpirun -np 2 ./bin/game_of_life_gpu_time 1

# Run on 4 nodes, 2 GPUs per node (8 GPUs total)
mpirun -np 4 ./bin/game_of_life_gpu_time 2
```

### Random Initialization

```bash
# Single GPU with random initialization
./bin/game_of_life_gpu_time 1 random

# 2 MPI ranks, 2 GPUs each, random initialization
mpirun -np 2 ./bin/game_of_life_gpu_time 2 random
```

## Command Line Arguments

1. `num_devices` - Number of GPUs to use per MPI rank (default: 1)
2. `init_mode` - Initialization mode: 'random' or 'predefined' (default: predefined)

## Output

The program reports (from rank 0 only):
- Total execution time in milliseconds
- Average time per timestep

Example output:
```
Execution completed
===================
Total time: 1234 ms
Average time per timestep: 1.234 ms
```

## Performance Notes

- MPI communication occurs only for boundary cells between ranks
- Each GPU processes a contiguous subregion for optimal memory access patterns
- CUDA constant memory is used for grid dimensions to minimize kernel overhead
- The framework automatically handles CUDA memory management and MPI synchronization

## Requirements

- CUDA Toolkit 11.0 or later
- GPU with compute capability 7.5 or higher (Turing, Ampere, or newer)
- MPI implementation (OpenMPI, MPICH, or similar)
- For multi-node execution: CUDA-aware MPI recommended for optimal performance
