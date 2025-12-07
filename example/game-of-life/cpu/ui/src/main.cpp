/**
 * @file main.cpp
 * @brief Game of Life example using the ParallelABM library.
 *
 * Demonstrates Conway's Game of Life simulation with ImGui-based
 * visualization using the ParallelABM library's CPU simulation capabilities.
 */

#include <ParallelABM/LocalEnvironmentCPU.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "GameOfLifeModel.h"
#include "GameOfLifeSimulation.h"
#include "GameOfLifeSpace.h"
#include "Renderer.h"

namespace {

/// Grid width
constexpr int kWidth = 100;

/// Grid height
constexpr int kHeight = 40;

/// Initial alive cell density (not used with deterministic patterns)
constexpr double kDensity = 0.3;

/// Number of timesteps
constexpr int kTimesteps = 1'000;

/// Frame delay in milliseconds
constexpr int kFrameDelayMs = 40;

/// Window width
constexpr int kWindowWidth = 1280;

/// Window height
constexpr int kWindowHeight = 720;

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
 * @brief Main entry point for the Game of Life simulation.
 *
 * Parses the number of threads from command line arguments,
 * initializes the simulation environment with deterministic patterns,
 * sets up the renderer, and runs the simulation with visualization.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code (0 on success)
 */
int main(int argc, char* argv[]) {
  try {
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

    // Create renderer and initialize rendering system
    Renderer renderer(kWidth, kHeight, kWindowWidth, kWindowHeight);
    if (!renderer.Setup()) {
      std::cerr << "Error: Failed to setup renderer.\n";
      return 1;
    }

    // Create local CPU environment with specified number of threads
    ParallelABM::LocalEnvironmentCPU environment(num_threads);

    // Create space and initialize with deterministic patterns
    auto space = std::make_unique<GameOfLifeSpace>(kWidth, kHeight, kDensity);
    space->Initialize();

    // Create the Game of Life model
    auto model = std::make_shared<GameOfLifeModel>(kWidth, kHeight);

    // Create simulation with rendering callback
    GameOfLifeSimulation simulation(argc, argv, std::move(space), model,
                                    environment, renderer,
                                    std::chrono::milliseconds(kFrameDelayMs));

    // Run the simulation for the specified number of timesteps
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
