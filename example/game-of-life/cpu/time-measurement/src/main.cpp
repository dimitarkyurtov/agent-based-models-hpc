/**
 * @file main.cpp
 * @brief Game of Life CPU time measurement example.
 *
 * Demonstrates Conway's Game of Life simulation using the ParallelABM library
 * for CPU performance measurement without visualization.
 */

#include <ParallelABM/LocalEnvironmentCPU.h>
#include <ParallelABM/Logger.h>
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "Cell.h"
#include "GameOfLifeModel.h"
#include "GameOfLifeSimulation.h"
#include "GameOfLifeSpace.h"

namespace {

/// Grid width
constexpr int kWidth = 1'000;

/// Grid height
constexpr int kHeight = 1'000;

/// Initial alive cell density
constexpr double kDensity = 0.3;

/// Number of timesteps
constexpr int kTimesteps = 1'000;

/// Default number of threads
constexpr int kDefaultThreads = 1;

/**
 * @brief Print usage information to stderr.
 * @param program_name Name of the executable
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [num_threads]\n"
            << "\n"
            << "Arguments:\n"
            << "  num_threads - Number of threads to use (default: "
            << kDefaultThreads << ")\n";
}

}  // namespace

/**
 * @brief Main entry point for Game of Life time measurement.
 *
 * Parses the number of threads from command line arguments,
 * initializes the simulation environment, runs the simulation,
 * and reports the execution time.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code (0 on success)
 */
int main(int argc, char* argv[]) {
  try {
    // Disable library logs by setting log level to Fatal
    ParallelABM::Logger::GetInstance().SetLogLevel(
        ParallelABM::LogLevel::kFatal);

    // Parse command line arguments for number of threads
    int num_threads = kDefaultThreads;

    if (argc > 1) {
      if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      num_threads = std::atoi(argv[1]);
    }

    // Validate number of threads
    if (num_threads <= 0) {
      std::cerr << "Error: Number of threads must be positive.\n";
      PrintUsage(argv[0]);
      return 1;
    }

    // Create local CPU environment with specified number of threads
    ParallelABM::LocalEnvironmentCPU environment(num_threads);

    // Create space and initialize with deterministic patterns
    auto space = std::make_unique<GameOfLifeSpace>(kWidth, kHeight, kDensity);
    space->Initialize();

    // Create the Game of Life model
    auto model = std::make_shared<GameOfLifeModel>(kWidth, kHeight);

    // Create simulation
    GameOfLifeSimulation simulation(argc, argv, std::move(space), model,
                                    environment);

    // Measure execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the simulation for the specified number of timesteps
    simulation.Start(kTimesteps);

    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print execution time only on rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);

      std::cout << "\nExecution completed\n"
                << "===================\n"
                << "Total time: " << duration.count() << " ms\n"
                << "Average time per timestep: "
                << (duration.count() / static_cast<double>(kTimesteps))
                << " ms\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Error: Unknown exception occurred.\n";
    return 1;
  }
}
