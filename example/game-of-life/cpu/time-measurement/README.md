# Game of Life CPU Time Measurement

This is a performance measurement application used to benchmark the ParallelABM library. It utilizes the CPU and MPI components of the library to perform ABM calculations. The application simulates Conway's Game of Life without visualization, focusing on measuring execution time and throughput.

## Purpose

This application is designed for:
- Benchmarking CPU performance with different thread counts
- Measuring speedup achieved through parallelization
- Performance analysis across different MPI process configurations
- Comparing execution times for various grid sizes and timesteps

## Building and Running

Build the project from the root directory:

```bash
mkdir build
cd build
cmake ..
make
```

Run the application with MPI processes and threads:

```bash
mpirun -n <num_processes> ./bin/game_of_life_cpu_time <num_threads>
```

Example usage:

```bash
# Single process, 4 threads
mpirun -n 1 ./bin/game_of_life_cpu_time 4

# 4 MPI processes, 2 threads each
mpirun -n 4 ./bin/game_of_life_cpu_time 2
```

## Output

The application prints timing information only from rank 0, including:
- Grid size and configuration parameters
- Total execution time in milliseconds
- Average time per timestep

All library logging is disabled to ensure clean performance measurements.
