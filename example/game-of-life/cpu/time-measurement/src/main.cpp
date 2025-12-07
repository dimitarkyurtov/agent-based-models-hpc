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
  std::cerr << "Usage: " << program_name << " [num_threads] [init_mode]\n"
            << "\n"
            << "Arguments:\n"
            << "  num_threads - Number of threads to use (default: "
            << kDefaultThreads << ")\n"
            << "  init_mode   - Initialization mode: 'random' or 'predefined' "
               "(default: predefined)\n";
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
    ParallelABM::Logger::GetInstance().SetLogLevel(
        ParallelABM::LogLevel::kFatal);

    int num_threads = kDefaultThreads;
    InitializationMode init_mode = InitializationMode::kPredefined;

    if (argc > 1) {
      if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      num_threads = std::atoi(argv[1]);
    }

    if (argc > 2) {
      const std::string mode_str = argv[2];
      if (mode_str == "random") {
        init_mode = InitializationMode::kRandom;
      } else if (mode_str == "predefined") {
        init_mode = InitializationMode::kPredefined;
      } else {
        std::cerr << "Error: Invalid initialization mode '" << mode_str
                  << "'. Must be 'random' or 'predefined'.\n";
        PrintUsage(argv[0]);
        return 1;
      }
    }

    if (num_threads <= 0) {
      std::cerr << "Error: Number of threads must be positive.\n";
      PrintUsage(argv[0]);
      return 1;
    }

    ParallelABM::LocalEnvironmentCPU environment(num_threads);

    auto space =
        std::make_unique<GameOfLifeSpace>(kWidth, kHeight, kDensity, init_mode);
    space->Initialize();

    auto model = std::make_shared<GameOfLifeModel>(kWidth, kHeight);

    GameOfLifeSimulation simulation(argc, argv, std::move(space), model,
                                    environment);

    auto start_time = std::chrono::high_resolution_clock::now();

    simulation.Start(kTimesteps);

    auto end_time = std::chrono::high_resolution_clock::now();

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
