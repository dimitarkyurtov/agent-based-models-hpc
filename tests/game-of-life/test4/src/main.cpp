/**
 * @file main.cpp
 * @brief Test 4: Game of Life with wide grid (800x400).
 *
 * Runs Conway's Game of Life simulation with an 800x400 grid configuration
 * for 100 timesteps with checkpoints every 10 steps.
 */

#include <ParallelABM/LocalEnvironment.h>
#include <ParallelABM/Logger.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "TestAgent.h"
#include "TestModel.h"
#include "TestSimulation.h"
#include "TestSpace.h"

namespace {

/// Grid width
constexpr int kWidth = 800;

/// Grid height
constexpr int kHeight = 400;

/// Number of timesteps
constexpr int kTimesteps = 100;

/// Checkpoint interval
constexpr int kCheckpointInterval = 10;

/// Default number of threads
constexpr int kDefaultThreads = 1;

/// Checkpoint file path
const std::string kCheckpointFile = "tests/game-of-life/checkpoints/test4.dat";

/**
 * @brief Print usage information to stderr.
 * @param program_name Name of the executable
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [num_threads] [output_file]\n"
            << "\n"
            << "Arguments:\n"
            << "  num_threads - Number of threads to use (default: "
            << kDefaultThreads << ")\n"
            << "  output_file - Output checkpoint file (default: "
            << kCheckpointFile << ")\n";
}

}  // namespace

/**
 * @brief Main entry point for Test 4.
 *
 * Initializes an 800x400 Game of Life grid, runs the simulation for 100
 * timesteps, and saves checkpoints every 10 steps to the tests/game-of-life/
 * checkpoints/test4.dat file.
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
    std::string output_file = kCheckpointFile;

    if (argc > 1) {
      if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      num_threads = std::atoi(argv[1]);
    }

    if (argc > 2) {
      output_file = argv[2];
    }

    if (num_threads <= 0) {
      std::cerr << "Error: Number of threads must be positive.\n";
      PrintUsage(argv[0]);
      return 1;
    }

    ParallelABM::LocalEnvironment environment(num_threads);

    // Initialize space with predefined patterns for deterministic results
    auto space = std::make_unique<TestSpace>(kWidth, kHeight);
    space->Initialize();

    auto model = std::make_shared<TestModel>(kWidth, kHeight);

    TestSimulation simulation(argc, argv, std::move(space), model, environment,
                              output_file, kCheckpointInterval);

    simulation.Start(kTimesteps);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Error: Unknown exception occurred.\n";
    return 1;
  }
}
