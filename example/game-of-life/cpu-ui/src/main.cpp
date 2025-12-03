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

/// Default grid width
constexpr int kDefaultWidth = 50;

/// Default grid height
constexpr int kDefaultHeight = 25;

/// Default initial alive cell density
constexpr double kDefaultDensity = 0.3;

/// Default number of timesteps
constexpr int kDefaultTimesteps = 100;

/// Frame delay in milliseconds
constexpr int kFrameDelayMs = 100;

/// Window width
constexpr int kWindowWidth = 1280;

/// Window height
constexpr int kWindowHeight = 720;

/**
 * @brief Print usage information to stderr.
 * @param program_name Name of the executable
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name
            << " [width] [height] [density] [timesteps]\n"
            << "\n"
            << "Arguments:\n"
            << "  width     - Grid width (default: " << kDefaultWidth << ")\n"
            << "  height    - Grid height (default: " << kDefaultHeight << ")\n"
            << "  density   - Initial alive cell density 0.0-1.0 (default: "
            << kDefaultDensity << ")\n"
            << "  timesteps - Number of simulation steps (default: "
            << kDefaultTimesteps << ")\n";
}

}  // namespace

/**
 * @brief Main entry point for the Game of Life simulation.
 *
 * Parses command line arguments, initializes the simulation environment,
 * creates the Game of Life model and space, sets up the renderer,
 * and runs the simulation with visualization.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code (0 on success)
 */
int main(int argc, char* argv[]) {
  try {
    // Parse command line arguments
    int width = kDefaultWidth;
    int height = kDefaultHeight;
    double density = kDefaultDensity;
    int timesteps = kDefaultTimesteps;

    if (argc > 1) {
      if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      width = std::atoi(argv[1]);
    }
    if (argc > 2) {
      height = std::atoi(argv[2]);
    }
    if (argc > 3) {
      density = std::atof(argv[3]);
    }
    if (argc > 4) {
      timesteps = std::atoi(argv[4]);
    }

    // Validate parameters
    if (width <= 0 || height <= 0) {
      std::cerr << "Error: Width and height must be positive.\n";
      PrintUsage(argv[0]);
      return 1;
    }

    if (density < 0.0 || density > 1.0) {
      std::cerr << "Error: Density must be between 0.0 and 1.0.\n";
      return 1;
    }

    if (timesteps <= 0) {
      std::cerr << "Error: Timesteps must be positive.\n";
      return 1;
    }

    // Create renderer and initialize rendering system
    Renderer renderer(width, height, kWindowWidth, kWindowHeight);
    if (!renderer.Setup()) {
      std::cerr << "Error: Failed to setup renderer.\n";
      return 1;
    }

    // Create local CPU environment with exactly 1 thread
    ParallelABM::LocalEnvironmentCPU environment(1);

    // Create space and initialize with random cells
    auto space = std::make_unique<GameOfLifeSpace>(width, height, density);
    space->Initialize();

    // Create the Game of Life model
    auto model = std::make_shared<GameOfLifeModel>(width, height);

    // Create simulation with rendering callback
    GameOfLifeSimulation simulation(argc, argv, std::move(space), model,
                                    environment, renderer,
                                    std::chrono::milliseconds(kFrameDelayMs));

    // Run the simulation for the specified number of timesteps
    simulation.Start(timesteps);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Error: Unknown exception occurred.\n";
    return 1;
  }
}
