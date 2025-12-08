/**
 * @file main.cpp
 * @brief Game of Life GPU time measurement example.
 *
 * Demonstrates Conway's Game of Life simulation using the ParallelABM library
 * for GPU performance measurement with CUDA acceleration.
 */

#include <ParallelABM/LocalEnvironment.h>
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

/// Default number of GPUs
constexpr int kDefaultGPUs = 1;

/**
 * @brief Print usage information to stderr.
 * @param program_name Name of the executable
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [num_devices] [init_mode]\n"
            << "\n"
            << "Arguments:\n"
            << "  num_devices - Number of GPU devices to use (default: "
            << kDefaultGPUs << ")\n"
            << "  init_mode   - Initialization mode: 'random' or 'predefined' "
               "(default: predefined)\n";
}

}  // namespace

/**
 * @brief Main entry point for Game of Life GPU time measurement.
 *
 * Parses the number of GPUs from command line arguments,
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

    int num_devices = kDefaultGPUs;
    InitializationMode init_mode = InitializationMode::kPredefined;

    if (argc > 1) {
      if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      num_devices = std::atoi(argv[1]);
    }

    if (argc > 2) {
      const std::string kModeStr = argv[2];
      if (kModeStr == "random") {
        init_mode = InitializationMode::kRandom;
      } else if (kModeStr == "predefined") {
        init_mode = InitializationMode::kPredefined;
      } else {
        std::cerr << "Error: Invalid initialization mode '" << kModeStr
                  << "'. Must be 'random' or 'predefined'.\n";
        PrintUsage(argv[0]);
        return 1;
      }
    }

    if (num_devices <= 0) {
      std::cerr << "Error: Number of devices must be positive.\n";
      PrintUsage(argv[0]);
      return 1;
    }

    // Create environment with 0 CPU cores (GPU-only) and specified GPU count
    ParallelABM::LocalEnvironment environment(
        1, static_cast<std::uint32_t>(num_devices));

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
                << (static_cast<double>(duration.count()) /
                    static_cast<double>(kTimesteps))
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
