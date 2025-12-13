# Agent-Based Models HPC

A C++ library that abstracts the complexity of running agent-based model simulations in high-performance computing (HPC) environments. This library provides a simplified interface for distributed agent-based simulations on CPU and GPU (Nvidia) architectures, handling the underlying parallelization and communication complexity.

### Examples

The repository includes a Game of Life implementation demonstrating the library's usage for building agent-based models. Both CPU and GPU example applications are provided for performance measurement and parallelism speedup analysis. Each example includes its own README with detailed usage instructions.

### Testing

The test suite runs Game of Life simulations against pre-recorded checkpoints taken every 50 steps. Each test verifies simulation correctness by comparing the hashed state of the entire simulation space at each checkpoint, ensuring the simulation produces expected results across different configurations.

### Docker Support

A Docker container is provided for building CUDA-enabled code on machines without NVIDIA GPUs or the CUDA toolkit installed locally. This enables development and compilation of GPU code without requiring local GPU hardware.

### AI-Assisted Development

This repository embraces AI-assisted development using Claude with sub-agents and MCP servers. The development workflow and tooling configuration can be explored in the repository's configuration files.

## Development

### Requirements

- CMake 3.14+
- C++17 compiler (GCC/Clang/MSVC)
- OpenMPI
- NVIDIA CUDA Toolkit (for GPU support)
- clang-format, clang-tidy

### Building

```bash
# Configure
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build

# Build with CUDA support
cmake -B build -DCUDA=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

### Running Tests

```bash
cd build && ctest --output-on-failure
```

### Code Formatting (Linter)

```bash
# Format all source files
find lib example tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) | xargs clang-format -i

# Check formatting without modifying files
find lib example tests -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) | xargs clang-format --dry-run --Werror
```

### Static Analysis

```bash
# Run clang-tidy on C++ sources
find lib example -type f \( -name '*.cpp' \) ! -path '*/external/*' ! -path '*/imgui*' ! -path '*/gpu/*' ! -name '*CUDA*' | xargs clang-tidy -p build
```
