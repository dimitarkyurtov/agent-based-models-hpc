# Agent-Based Models HPC

A C++ library that abstracts the complexity of running agent-based model simulations in an HPC environment on Nvidia GPUs.

## Project Structure

```
.
├── lib/                    # Static library source code
│   ├── include/            # Public headers
│   └── src/               # Implementation files
├── example/               # Example application
├── tests/                 # Unit tests (Google Test)
├── .clang-format         # Code formatting configuration (Google style)
├── .clang-tidy           # Static analysis configuration
├── .githooks/            # Shared git hooks (pre-push checks)
└── .github/workflows/    # CI/CD workflows
```

## Features

- **Static C++ Library**: Core library for agent-based model simulations
- **Example Application**: Demonstrates library usage
- **Unit Tests**: Comprehensive test suite using Google Test
- **Code Quality Tools**:
  - clang-format for code formatting (Google style)
  - clang-tidy for static analysis
- **CI/CD**: GitHub Actions workflow for automated testing and quality checks

## Requirements

- CMake 3.14 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
- clang-format (for code formatting)
- clang-tidy (for static analysis)
- OpenMPI (for MPI support)
- NVIDIA CUDA Toolkit (for GPU support)

### Docker (Recommended for HPC Development)

The easiest way to get started with all dependencies is to use Docker:

**Prerequisites:**
- Docker Engine 19.03+
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA drivers on host machine

**Installing NVIDIA Docker Runtime:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Using Docker Compose:**
```bash
# Build and start the container
docker-compose up -d

# Enter the container
docker-compose exec mpi-nvidia /bin/bash

# Stop the container
docker-compose down
```

**Or use the convenience script:**
```bash
./docker-run.sh
```

The Docker container includes:
- Ubuntu 22.04 LTS
- NVIDIA CUDA 12.3.1
- OpenMPI 4.1.6 with CUDA support
- CMake, clang, clang-format, clang-tidy
- Development tools (gdb, valgrind)

### Installing Dependencies (Native)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build clang-format clang-tidy \
  libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install cmake llvm open-mpi
```

**Windows:**
- Install Visual Studio with C++ support
- Install CMake from https://cmake.org/download/
- Install Microsoft MPI

## Building

### Quick Start

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Run example application
./build/example/example_app
```

### Build Options

- `BUILD_TESTING`: Enable/disable tests (default: ON)

```bash
cmake -B build -DBUILD_TESTING=OFF
```

## Running Tests

```bash
# Run all tests
cd build && ctest

# Run tests with verbose output
cd build && ctest --verbose

# Run tests with output on failure
cd build && ctest --output-on-failure

# Run specific test
./build/tests/mylib_tests --gtest_filter=CalculatorTest.AddPositiveNumbers
```

## Code Formatting

The project uses Google C++ style. Format your code before committing:

```bash
# Format a single file
clang-format -i lib/src/calculator.cpp

# Format all source files
find lib example tests -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Check formatting without modifying files
find lib example tests -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror
```

## Static Analysis

Run clang-tidy for static analysis:

```bash
# Configure with compile commands
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run clang-tidy on library sources
find lib -name '*.cpp' | xargs clang-tidy -p build

# Run clang-tidy on all sources
find lib example -name '*.cpp' | xargs clang-tidy -p build
```

## Git Hooks

The project includes git hooks to ensure code quality before pushing. These hooks automatically run formatting checks, linting, and tests.

### Setup

After cloning the repository, configure git to use the shared hooks:

```bash
git config core.hooksPath .githooks
```

This needs to be done only once per clone.

### Pre-push Hook

The pre-push hook runs automatically before every push and performs:
1. Code formatting verification (clang-format)
2. Static analysis (clang-tidy)
3. Unit tests (ctest)

If any check fails, the push will be blocked until the issues are fixed.

### Manual Testing

To test the pre-push hook without actually pushing:

```bash
.githooks/pre-push
```

## CI/CD

The project uses GitHub Actions for continuous integration. The CI pipeline runs entirely inside Docker containers on Ubuntu latest:

1. **Docker Build**: Builds the Docker image with all dependencies (MPI, CUDA, development tools)
2. **Format Check**: Verifies code formatting against Google style using clang-format
3. **Static Analysis**: Runs clang-tidy with warnings-as-errors for code quality checks
4. **Application Build**: Builds the project inside the Docker container using CMake
5. **Verification**: Checks that all build artifacts were created successfully

The CI runs automatically on:
- Pushes to `main` and `develop` branches
- Pull requests targeting `main` and `develop` branches

Note: The CI only builds the application; it does not run tests or execute the compiled binaries. GPU runtime is not available in the CI environment.

## Library Usage

### In Your CMake Project

```cmake
# Add the library
add_subdirectory(path/to/agent-based-models-hpc)

# Link against your target
target_link_libraries(your_target PRIVATE mylib)
```

### In Your Code

```cpp
#include "calculator.h"

int main() {
  mylib::Calculator calc;

  int result = calc.Add(10, 5);
  std::cout << "10 + 5 = " << result << std::endl;

  return 0;
}
```

## Contributing

1. Fork the repository
2. Clone and set up git hooks: `git config core.hooksPath .githooks`
3. Create a feature branch
4. Make your changes
5. Format your code: `clang-format -i <files>`
6. Run tests: `cd build && ctest`
7. Run static analysis: `clang-tidy <files> -p build`
8. Submit a pull request (pre-push hooks will run automatically)

## License

This project is provided as-is for educational purposes.
