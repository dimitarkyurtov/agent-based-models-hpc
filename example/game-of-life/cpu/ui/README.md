# Example Application

This is an example application with UI used to verify the correctness of the ParallelABM library. It utilizes the CPU and MPI components of the library in order to do the ABM calculations. The application simulates Conway's Game of Life and uses ImGui for rendering the simulation space.

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
mpirun -n <num_processes> ./bin/game_of_life_cpu_ui <num_threads> [init_mode]
```

### Arguments

- `num_threads` - Number of CPU threads per MPI process (default: 1)
- `init_mode` - Grid initialization mode (default: predefined)
  - `random` - Random cell initialization based on density (30%)
  - `predefined` - Deterministic patterns (gliders, oscillators, etc.)

### Examples

```bash
# Single process, 1 thread, predefined patterns (default)
mpirun -n 1 ./bin/game_of_life_cpu_ui

# Single process, 4 threads, predefined patterns
mpirun -n 1 ./bin/game_of_life_cpu_ui 4

# Single process, 4 threads, random initialization
mpirun -n 1 ./bin/game_of_life_cpu_ui 4 random

# 2 MPI processes, 2 threads each, predefined patterns
mpirun -n 2 ./bin/game_of_life_cpu_ui 2 predefined
```

The simulation can be executed with multiple MPI processes and threads. Only the coordinator process performs rendering on each time step.
