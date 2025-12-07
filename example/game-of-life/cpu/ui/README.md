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
mpirun -n <num_processes> ./bin/game_of_life_cpu_ui <num_threads>
```

The simulation can be executed with multiple MPI processes and threads. Only the coordinator process performs rendering on each time step.
